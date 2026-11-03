"""Evaluate a chain on a split: writes detection predictions (preds/<token>.npz) and a report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qdistill import metrics as M
from qdistill.np_io import load_npz
from qdistill.light_blocks import install_light_voxel, load_light_voxel
from qdistill.data import IMAGE_FORMATS, load_frame_images, split_tokens
from qdistill.student import TapNormalizer, TapStudent
from qdistill.teacher import load_head, load_processor, run_head
from qdistill.trt_install import TrtHead
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]




def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="mini_val")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-mini.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-mini")
    p.add_argument("--resized", type=Path, default=ROOT / "data/resized/v1.0-mini")
    p.add_argument("--images", choices=IMAGE_FORMATS, default="jpeg_gpu",
                   help="jpeg_gpu, jpeg or pre-resized arrays")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-mini")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--light-voxel", type=Path, default=None,
                   help="Stage 2: distilled voxel-conv block checkpoint in place of the head's conv stack")
    p.add_argument("--trt-voxel", type=Path, default=None,
                   help="Stage 2: TensorRT fp16 engine of the voxel-conv block (exact stack or a distilled one)")
    p.add_argument("--chain", default=None,
                   help="zero-shot head ablation spec (qdistill.head_chain: prior:zero|mean|noise=r, "
                        "skip_sa:<l>, skip_ca:<l>, comma-separated)")
    p.add_argument("--stride", type=int, default=1, help="every n-th token of --split")
    p.add_argument("--merged-pool", action="store_true",
                   help="merged-camera voxel pooling straight into the voxel engine (bit-identical input)")
    p.add_argument("--trt-refiner", type=Path, default=None,
                   help="Stage 2: TensorRT fp16 engine of the occupancy refiner (trt_occ_refiner.py)")
    p.add_argument("--trt-student", type=Path, default=None,
                   help="fused student + tap de-normalisation + LLM adaptor TensorRT engine (trt_student_fpn.py) "
                        "in place of the PyTorch student and the head's adaptor")
    args = p.parse_args()
    (args.output / "preds").mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    stats_file = train_args.get("tap_stats") or args.cache / f"tap_stats_{train_args['train_split'].replace(':', '_')}.npz"
    stats = dict(np.load(stats_file))
    student = TapStudent(train_args["backbone"], pretrained=False).cuda().eval()
    student.load_state_dict(ckpt["model"])
    norm = TapNormalizer(stats).cuda()
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
    processor = load_processor(args.weights)



    tokens = split_tokens(args.splits, args.split)
    tokens = tokens[:: args.stride]
    agree = {"occ": np.zeros((len(M.OCC_CLASSES), 2)), "map": np.zeros((len(M.MAP_CLASSES), 2)), "det": np.zeros((7, 3))}
    vs_gt = {"student": np.zeros((7, 3)), "teacher": np.zeros((7, 3))}
    t_student, t_head = [], []
    for token in tokens:
        frame = PerceptionFrame(args.packed / token)
        _, img_metas = processor(frame)
        images = load_frame_images(args.packed if args.images.startswith("jpeg") else args.resized, token, args.images)[None].cuda()
        torch.cuda.synchronize()
        t = time.perf_counter()
        if sfpn is not None:  # the engine's vit tap; llm is a stand-in (the adaptor reads the engine's levels)
            with torch.no_grad():
                vit, llm = sfpn(images)
            torch.cuda.synchronize()
            t_student.append(time.perf_counter() - t)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vit_n, llm_n = student(images)
            torch.cuda.synchronize()
            t_student.append(time.perf_counter() - t)
            vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])

        t = time.perf_counter()
        pred = run_head(head, vit, llm, img_metas)
        torch.cuda.synchronize()
        t_head.append(time.perf_counter() - t)
        np.savez(args.output / "preds" / f"{token}.npz", boxes=pred["boxes"], scores=pred["scores"], labels=pred["labels"])

        s_boxes, s_labels = M.pred_to_ego(pred, frame.lidar2ego)
        g_boxes, g_labels = M.filter_gt(frame.gt)
        vs_gt["student"] += M.det_counts(s_boxes, s_labels, g_boxes, g_labels)
        path = args.cache / "head" / f"{token}.npz"  # teacher outputs: optional (scripts/backbone/cache_teacher.py)
        if not path.exists():
            continue
        teacher = dict(load_npz(path))
        agree["occ"] += M.iou_counts(pred["occ"], teacher["occ"], M.OCC_CLASSES)
        agree["map"] += M.iou_counts(pred["map"], teacher["map"], M.MAP_CLASSES)
        t_boxes, t_labels = M.pred_to_ego(teacher, frame.lidar2ego)
        agree["det"] += M.det_counts(s_boxes, s_labels, t_boxes, t_labels)
        vs_gt["teacher"] += M.det_counts(t_boxes, t_labels, g_boxes, g_labels)

    per_class = {
        who: {DET_CLASS_NAMES[k]: {**M.prf(c[k]), "n_gt": int(c[k, 2])} for k in range(7) if c[k, 2] > 0}
        for who, c in vs_gt.items()
    }
    n_student = sum(q.numel() for q in student.parameters())
    n_head = sum(q.numel() for q in head.parameters())
    report = {
        "ckpt": str(args.ckpt),
        "epoch": ckpt.get("epoch"),
        "split": args.split,
        "frames": len(tokens),
        "agreement_with_teacher": {
            "occ_miou": M.miou(agree["occ"]), "map_miou": M.miou(agree["map"]), **{f"det_{k}": v for k, v in M.prf(agree["det"].sum(0)).items()},
        },
        "det_vs_gt": {who: M.prf(c.sum(0)) for who, c in vs_gt.items()},
        "det_vs_gt_per_class": per_class,
        "params_M": {"student": n_student / 1e6, "head": n_head / 1e6, "teacher_vlm": 4539.0},
        "latency_ms_median": {"student_bf16": 1000 * float(np.median(t_student[1:])), "head_bf16": 1000 * float(np.median(t_head[1:]))},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "det_vs_gt_per_class"}, indent=2))
    print(f"{'class':15s} {'student F1':>10s} {'teacher F1':>10s} {'n_gt':>6s}")
    for name, st in per_class["student"].items():
        tf = per_class["teacher"].get(name, {}).get("f1", float("nan"))
        print(f"{name:15s} {st['f1']:10.3f} {tf:10.3f} {st['n_gt']:6d}")


if __name__ == "__main__":
    main()
