"""Dump map logits and occupancy labels of a chain, for GT scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qdistill.np_io import load_npz
from qdistill.light_blocks import install_light_voxel, load_light_voxel
from qdistill.data import IMAGE_FORMATS, load_frame_images, split_tokens
from qdistill.student import TapNormalizer, TapStudent
from qdistill.teacher import load_head, load_processor, run_head
from qdistill.trt_install import TrtHead
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("SS", "TS", "ST")




def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/dumps")
    p.add_argument("--variants", nargs="+", default=["SS"], choices=VARIANTS)
    p.add_argument("--split", default="val_quick")
    p.add_argument("--stride", type=int, default=3, help="keep every n-th frame of the split (all scenes kept)")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--images", choices=IMAGE_FORMATS, default="jpeg_gpu", help="jpeg_gpu: synced GPU decode")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--light-voxel", type=Path, default=None,
                   help="Stage 2: distilled voxel-conv block checkpoint in place of the head's conv stack")
    p.add_argument("--trt-voxel", type=Path, default=None,
                   help="Stage 2: TensorRT fp16 engine of the voxel-conv block (exact stack or a distilled one)")
    p.add_argument("--trt-refiner", type=Path, default=None,
                   help="Stage 2: TensorRT fp16 engine of the occupancy refiner (trt_occ_refiner.py)")
    p.add_argument("--chain", default=None,
                   help="zero-shot head ablation spec (qdistill.head_chain: prior:zero|mean|noise=r, "
                        "skip_sa:<l>, skip_ca:<l>, comma-separated)")
    p.add_argument("--merged-pool", action="store_true",
                   help="merged-camera voxel pooling straight into the voxel engine (bit-identical input)")
    p.add_argument("--trt-student", type=Path, default=None,
                   help="fused student + tap de-normalisation + LLM adaptor TensorRT engine (trt_student_fpn.py) "
                        "in place of the PyTorch student and the head's adaptor")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    stats_file = train_args.get("tap_stats") or args.cache / f"tap_stats_{train_args['train_split'].replace(':', '_')}.npz"
    student = TapStudent(train_args["backbone"], pretrained=False).cuda().eval()
    student.load_state_dict(ckpt["model"])
    norm = TapNormalizer(dict(np.load(stats_file))).cuda()
    head = load_head(args.weights)
    if args.light_voxel is not None:  # stays installed for the whole run
        light = install_light_voxel(head, load_light_voxel(args.light_voxel))
        light.__enter__()
    if args.trt_voxel is not None or args.trt_refiner is not None:  # installed for the whole run
        if args.trt_voxel is not None and args.light_voxel is not None:
            raise SystemExit("--trt-voxel replaces the voxel block: drop --light-voxel")
        trt_head = TrtHead(head, args.trt_voxel, args.trt_refiner)
        print("TensorRT engines installed:", trt_head.info(), flush=True)
        if args.merged_pool:
            from qdistill import merged_pool
            merged_pool.install(head, trt_head)
            print("merged-camera voxel pooling installed", flush=True)
    if args.chain:
        from qdistill import head_chain
        print("head ablation installed:", head_chain.install(head, args.chain), flush=True)
    sfpn = None
    if args.trt_student is not None:
        from qdistill import trt_student
        sfpn = trt_student.load(args.trt_student, head)
        print(f"student engine installed ({type(sfpn).__name__}):", sfpn.info(), flush=True)
    assert sfpn is None or list(args.variants) == ["SS"], "--trt-student: student taps only (variant SS)"
    processor = load_processor(args.weights)
    rig = None

    tokens = split_tokens(args.splits, args.split)[:: args.stride]
    outdirs = {v: args.output / f"{args.label}_{v}" for v in args.variants}
    for d in outdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    need_teacher = any("T" in v for v in args.variants)
    for i, token in enumerate(tokens):
        if all((d / f"{token}.npz").exists() for d in outdirs.values()):
            continue
        frame = PerceptionFrame(args.packed / token)
        _, img_metas = processor(frame)
        if rig is not None:
            img_metas = rig.apply(img_metas, token, frame.cam_intrinsic)
        images = load_frame_images(args.packed, token, args.images)[None].cuda()
        if sfpn is not None:  # engine vit tap + llm stand-in (the adaptor reads the engine's levels)
            with torch.no_grad():
                vit, llm = sfpn(images)
            taps = {"S": {"vit": vit, "llm": llm}}
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vit_n, llm_n = student(images)
            taps = {"S": {"vit": norm.denormalize("vit", vit_n[0]), "llm": norm.denormalize("llm", llm_n[0])}}
        if need_teacher:
            cached = load_npz(args.cache / "taps" / f"{token}.npz")
            taps["T"] = {k: torch.from_numpy(cached[k]).cuda() for k in ("vit", "llm")}
        for v, d in outdirs.items():
            pred = run_head(head, taps[v[0]]["vit"], taps[v[1]]["llm"], img_metas)
            np.savez_compressed(d / f"{token}.npz", map_logits=pred["map_logits"], occ=pred["occ"])
        print(f"{i + 1}/{len(tokens)} {token}", flush=True)
    print(f"map dump {args.label} done ({len(tokens)} frames, variants {' '.join(args.variants)})")


if __name__ == "__main__":
    main()
