"""Export, build and check the BEV encoder TensorRT engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qdistill.student_io import load_student
from qdistill import head_chain, merged_pool
from qdistill.data import load_frame_images, split_tokens
from qdistill.teacher import load_head, load_processor
from qdistill.trt_encoder import NoCAEncoder, TrtEncoder, build, export_onnx
from qdistill.trt_install import TrtHead
from qdistill.trt_utils import median_ms
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]
DENSE_AB = "skip_ca:0,skip_ca:1,skip_ca:2,skip_ca:3,skip_ca:4,skip_ca:5,enc_ckpt:runs/ft_noca/final.pt"


def err(x: torch.Tensor, ref: torch.Tensor) -> dict:
    x, ref = x.float().reshape(-1, ref.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    cos = torch.nn.functional.cosine_similarity(x, ref, dim=-1)
    return {"nmse": float(((x - ref) ** 2).mean() / ref.var()), "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
            "max_abs": float((x - ref).abs().max()), "nonfinite": int((~torch.isfinite(x)).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("export", "build", "check"), required=True)
    p.add_argument("--chain", default=DENSE_AB, help="head_chain spec giving the deployed encoder")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/e079")
    p.add_argument("--no-ln-fp32", action="store_true")
    p.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--enc-fp32", type=Path, default=None,
                   help="encoder layers from this train_enc_recover final.pt in fp32 (export / check reference)")
    p.add_argument("--tag", default="", help="engine name suffix (e.g. _lnfp16)")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / "enc_noca.onnx"
    eng = args.output / f"enc_noca_fp16{args.tag}.engine"

    if args.phase == "build":
        r = build(onnx_path, eng, ln_fp32=not args.no_ln_fp32, precision=args.precision)
        print(json.dumps(r), flush=True)
        return

    head = load_head(args.weights)
    if args.phase == "export":  # merged pooling first, as check
        merged_pool.install(head, TrtHead(head, args.voxel_engine, args.refiner_engine))
        head_chain.install(head, args.chain)
        export_onnx(NoCAEncoder(head, args.enc_fp32), onnx_path)
        print(f"export ok ({onnx_path.stat().st_size / 2**20:.0f} MB) -> {onnx_path}", flush=True)
        return

    # check: capture the deployed encoder's inputs / output on real frames (ablation after merged pooling, as
    # eval_stage1: warp_grid replaces the pooling it installs)
    student, norm = load_student(args.ckpt, args.cache)
    trt_head = TrtHead(head, args.voxel_engine, args.refiner_engine)
    merged_pool.install(head, trt_head)
    head_chain.install(head, args.chain)
    bev = head.bev_modeling
    tr = bev.head.transformer
    processor = load_processor(args.weights)
    caps = []
    f_gbf = tr.get_bev_features

    def capture(mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=None, **kw):
        out = f_gbf(mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=bev_pos, **kw)
        caps.append({"q": bev_queries.detach().reshape(1, -1, 256).clone(), "pos": bev_pos.flatten(2).permute(0, 2, 1).clone(),
                     "bf16": out.detach().clone(), "args": (mlvl_feats, bev_queries, bev_h, bev_w, bev_pos, kw)})
        return out
    tr.get_bev_features = capture
    tokens = split_tokens(args.splits, "val_quick")[::48][: args.frames]
    for t in tokens:
        metas = processor(PerceptionFrame(args.packed / t))[1]
        images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])
        with torch.no_grad():
            bev(img_vit_feats=vit.to(torch.bfloat16), img_llm_feats=llm.to(torch.bfloat16), img_metas=[metas])
        caps[-1]["args"] = None if len(caps) > 1 else caps[-1]["args"]  # keep one frame's call for the latency run
    tr.get_bev_features = f_gbf
    print(f"captured {len(caps)} frames; query {tuple(caps[0]['q'].shape)} {caps[0]['q'].dtype}, "
          f"out {tuple(caps[0]['bf16'].shape)}", flush=True)

    ref_enc = NoCAEncoder(head, args.enc_fp32).cuda().eval()
    run = TrtEncoder(eng, out_dtype=torch.float32)
    rows = {"bf16": [], "trt": []}
    with torch.no_grad():
        for c in caps:
            ref = ref_enc(c["q"].float(), c["pos"].float())
            rows["bf16"].append(err(c["bf16"], ref))
            t_out = run(c["q"], c["pos"])
            rows["trt"].append(err(t_out, ref))
            rows.setdefault("trt_vs_deployed_bf16", []).append(err(t_out, c["bf16"]))
    summ = {k: {m: (float(np.max([r[m] for r in v])) if m in ("nmse", "max_abs", "nonfinite") else float(np.min([r[m] for r in v]))
                    if m == "cos_min" else float(np.mean([r[m] for r in v]))) for m in v[0]} for k, v in rows.items()}
    print("error vs fp32 reference (worst over frames):", json.dumps(summ, indent=1), flush=True)

    a = caps[0]["args"]
    # the upstream encoder itself in fp32 (MSDA via the upstream PyTorch fallback, fp32): validates ref32 (merged TSA,
    # layer order) and gives the bf16 path's error against exact upstream math
    import copy
    from qwen_drive_perception import attention
    f_msda = attention.multi_scale_deformable_attn_cuda
    attention.multi_scale_deformable_attn_cuda = lambda v, ss, lsi, loc, w, im2col_step: \
        attention.multi_scale_deformable_attn_pytorch(v.float(), ss, loc.float(), w.float())
    enc_bf16 = tr.encoder
    tr.encoder = copy.deepcopy(enc_bf16).float()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        up32 = f_gbf([m.float() for m in a[0]], a[1].float(), a[2], a[3], bev_pos=a[4].float(), **a[5])
        ref0 = ref_enc(caps[0]["q"].float(), caps[0]["pos"].float())
    tr.encoder, attention.multi_scale_deformable_attn_cuda = enc_bf16, f_msda
    up = {"ref32_vs_upstream_fp32": err(ref0, up32), "bf16_vs_upstream_fp32": err(caps[0]["bf16"], up32),
          "trt_vs_upstream_fp32": err(run(caps[0]["q"], caps[0]["pos"]), up32)}
    print("frame 0 vs the upstream encoder in fp32:", json.dumps(up, indent=1), flush=True)
    with torch.no_grad():
        pt = median_ms(lambda: f_gbf(a[0], a[1], a[2], a[3], bev_pos=a[4], **a[5]), n=50)
        te = median_ms(lambda: run(caps[0]["q"], caps[0]["pos"]), n=50)
    out = {"frames": len(caps), "ablate": args.chain, "engine": str(eng), "scratch_MiB": run.scratch_MiB,
           "error_vs_fp32_ref": summ, "frame0_vs_upstream_fp32": up, "per_frame": rows,
           "latency_ms": {"pytorch_bf16_get_bev_features": pt, "trt_engine": te}}
    (args.output / f"trt_bev_encoder_check{args.tag}.json").write_text(json.dumps(out, indent=1))
    print("latency (median, p90 ms):", json.dumps(out["latency_ms"]), flush=True)


if __name__ == "__main__":
    main()
