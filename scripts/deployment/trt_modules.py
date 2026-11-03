"""Export and build small head modules as TensorRT engines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qdistill.student_io import load_student
from qdistill import head_chain, merged_pool, trt_modules
from qdistill.data import load_frame_images, split_tokens
from qdistill.teacher import load_head, load_processor
from qdistill.trt_encoder import build
from qdistill.trt_install import TrtHead
from qdistill.trt_utils import median_ms
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]
CHAIN = ("skip_ca:0,skip_ca:1,skip_ca:2,skip_ca:3,skip_ca:4,skip_ca:5,enc_ckpt:runs/ft_fp32/final.pt,"
         "enc_trt:outputs/engines/enc_noca_fp16.engine,det_sample:runs/sampler/final.pt")


def err(x: torch.Tensor, ref: torch.Tensor) -> dict:
    x, ref = x.float().reshape(-1, ref.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    cos = torch.nn.functional.cosine_similarity(x, ref, dim=-1)
    return {"nmse": float(((x - ref) ** 2).mean() / ref.var()), "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
            "max_abs": float((x - ref).abs().max()), "nonfinite": int((~torch.isfinite(x)).sum())}


def worst(rows: list[dict]) -> dict:
    return {m: float(np.min([r[m] for r in rows])) if m == "cos_min" else float(np.mean([r[m] for r in rows]))
            if m == "cos_mean" else float(np.max([r[m] for r in rows])) for m in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", default=",".join(trt_modules.TARGETS), help="comma list of trt_modules.TARGETS")
    p.add_argument("--chain", default=CHAIN, help="head_chain spec of the deployed chain")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/d050_modules")
    p.add_argument("--heads-ckpt", type=Path, default=None,
                   help="train_enc_recover --train-heads final.pt; its fp32 head masters are exported "
                        "(and loaded, rounded, into the eager reference)")
    p.add_argument("--frames", type=int, default=4, help="the occ-fuse input is ~330 MB per frame (kept on the CPU)")
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.names.split(",")

    head = load_head(args.weights)
    head_chain.install(head, args.chain)
    masters = {}
    if args.heads_ckpt is not None:
        trt_modules.load_heads(head, args.heads_ckpt)
        masters = torch.load(args.heads_ckpt, map_location="cpu", weights_only=False)["heads"]
    student, norm = load_student(args.ckpt, args.cache)
    trt_head = TrtHead(head, args.voxel_engine, args.refiner_engine)
    merged_pool.install(head, trt_head)
    bev = head.bev_modeling
    processor = load_processor(args.weights)

    caps = {n: [] for n in names}
    hooks = []
    for n in names:
        m = trt_modules.get(bev, trt_modules.TARGETS[n])

        def hook(mod, a, out, n=n):
            y = out[0] if isinstance(out, (list, tuple)) else out
            caps[n].append({"x": a[0].detach().cpu(), "y": y.detach().cpu()})
        hooks.append(m.register_forward_hook(hook))
    tokens = split_tokens(args.splits, "val_quick")[::48][: args.frames]
    for t in tokens:
        metas = processor(PerceptionFrame(args.packed / t))[1]
        images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])
        with torch.no_grad():
            bev(img_vit_feats=vit.to(torch.bfloat16), img_llm_feats=llm.to(torch.bfloat16), img_metas=[metas])
    for h in hooks:
        h.remove()
    del student
    torch.cuda.empty_cache()
    print("captured:", {n: (len(c), tuple(c[0]["x"].shape) if c else None, c[0]["x"].dtype if c else None)
                        for n, c in caps.items()}, flush=True)

    # merged into an existing report: the queue runs one module per process
    rpath = args.output / "trt_modules_check.json"
    report = json.loads(rpath.read_text()) if rpath.exists() else {"modules": {}}
    report.update({"frames": len(tokens), "ablate": args.chain, "heads_ckpt": str(args.heads_ckpt)})
    for n in names:
        if not caps[n]:
            report["modules"][n] = {"skipped": "not called in this pipeline"}
            print(n, "not called: skipped", flush=True)
            continue
        assert all(c["x"].shape == caps[n][0]["x"].shape for c in caps[n]), f"{n}: input shape varies across frames"
        mod = trt_modules.get(bev, trt_modules.TARGETS[n])
        onnx_path, eng = args.output / f"{n}.onnx", trt_modules.engine_path(args.output, n)
        trt_modules.export_onnx(mod, n, caps[n][0]["x"], onnx_path, sd=masters.get(n))
        b = build(onnx_path, eng, ln_fp32=True)
        if b.get("failed"):
            report["modules"][n] = {"build": b}
            print(n, "engine build FAILED", flush=True)
            continue
        ref32 = trt_modules.fp32_copy(mod, masters.get(n)).cuda()
        if n in trt_modules.LIST_OUT:
            ref32 = trt_modules._Single(ref32)
        run = trt_modules.ModEngine(eng)
        rows = {"bf16": [], "trt": [], "trt_vs_deployed_bf16": []}
        amax = {"in": 0.0, "out_fp32": 0.0}
        with torch.no_grad():
            for c in caps[n]:
                x = c["x"].cuda()
                r = ref32(x.float())
                t = run(x)
                rows["bf16"].append(err(c["y"].cuda(), r))
                rows["trt"].append(err(t, r))
                rows["trt_vs_deployed_bf16"].append(err(t, c["y"].cuda()))
                amax["in"] = max(amax["in"], float(x.float().abs().max()))
                amax["out_fp32"] = max(amax["out_fp32"], float(r.abs().max()))
            x = caps[n][0]["x"].cuda()
            lat = {"eager_bf16": median_ms(lambda: mod(x), n=50), "trt": median_ms(lambda: run(x), n=50)}
        report["modules"][n] = {"in_shape": list(caps[n][0]["x"].shape), "build": b, "scratch_MiB": run.scratch_MiB,
                                "error_vs_fp32": {k: worst(v) for k, v in rows.items()}, "amax": amax, "latency_ms": lat}
        print(n, json.dumps(report["modules"][n]["error_vs_fp32"]), "amax", amax, "latency", lat, flush=True)
        del run, ref32
        torch.cuda.empty_cache()
    rpath.write_text(json.dumps(report, indent=1))
    print("->", rpath, flush=True)


if __name__ == "__main__":
    main()
