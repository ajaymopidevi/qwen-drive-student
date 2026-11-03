"""Export, build and check the fused occupancy + map TensorRT engine."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

from qdistill import dense_engine, head_chain, merged_pool, trt_modules
from qdistill.data import load_frame_images, split_tokens
from qdistill.teacher import load_head, load_processor, run_head
from qdistill.trt_encoder import build
from qdistill.trt_install import TrtHead
from qdistill.trt_utils import median_ms
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]


def err(x: torch.Tensor, ref: torch.Tensor, dim: int) -> dict:
    x, ref = x.float(), ref.float()
    return {"nmse": float(((x - ref) ** 2).mean() / ref.var()), "max_abs": float((x - ref).abs().max()),
            "label_agree": float((x.argmax(dim) == ref.argmax(dim)).float().mean()),
            "nonfinite": int((~torch.isfinite(x)).sum())}


def worst(rows: list[dict]) -> dict:
    return {m: float(np.min([r[m] for r in rows])) if m == "label_agree" else float(np.max([r[m] for r in rows])) for m in rows[0]}


def chain(args, head):
    trt_head = TrtHead(head, args.voxel_engine, args.refiner_engine)
    merged_pool.install(head, trt_head)
    print("chain:", head_chain.install(head, args.chain), file=sys.stderr, flush=True)
    return trt_head


def phase_export(args):
    from qdistill.student_io import load_student

    head = load_head(args.weights)
    bev = head.bev_modeling
    defs = {n: copy.deepcopy(trt_modules.get(bev, trt_modules.TARGETS[n])).cpu() for n in dense_engine.NAMES}
    trt_head = chain(args, head)  # noqa: F841 (keeps the engines alive)
    hd, tr = bev.head, bev.head.transformer
    student, norm = load_student(args.ckpt, args.cache)
    processor = load_processor(args.weights)
    cur, caps = {}, []
    f_ab, f_av = tr._adapt_bev_for_occ, tr._adapt_volume_for_occ

    def ab(bev_feat, *a, **kw):
        cur["bev"] = bev_feat.detach().cpu()
        return f_ab(bev_feat, *a, **kw)

    def av(volume_feat, *a, **kw):  # also called inside _adapt_bev_for_occ (the 16 x 16 BEV view): keep the Z8 volume
        if volume_feat.shape[2] == 8 and volume_feat.shape[1] != 16:
            cur["vol"] = volume_feat.detach().cpu()
        return f_av(volume_feat, *a, **kw)
    tr._adapt_bev_for_occ, tr._adapt_volume_for_occ = ab, av
    h1 = tr.occ_pred_head.register_forward_hook(lambda m, a, o: cur.__setitem__("occ", o.detach().cpu()))
    h2 = hd.seg_decoder.register_forward_hook(lambda m, a, o: cur.__setitem__("seg", o.detach().cpu()))
    metas0 = None
    for t in split_tokens(args.splits, "val_quick")[::48][: args.frames]:
        metas = processor(PerceptionFrame(args.packed / t))[1]
        metas0 = metas0 or metas
        images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])
        with torch.no_grad():
            bev(img_vit_feats=vit.to(torch.bfloat16), img_llm_feats=llm.to(torch.bfloat16), img_metas=[metas])
        assert set(cur) == {"bev", "vol", "occ", "seg"}, f"capture incomplete: {sorted(cur)}"
        caps.append(dict(cur))
        cur.clear()
    h1.remove()
    h2.remove()
    tr._adapt_bev_for_occ, tr._adapt_volume_for_occ = f_ab, f_av
    grids = dense_engine.capture_grids(head, metas0)
    print("captured", len(caps), {k: tuple(v.shape) for k, v in caps[0].items()}, "grids",
          {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in grids.items()}, flush=True)
    del student, head, trt_head, bev, hd, tr
    gc.collect()
    torch.cuda.empty_cache()

    model = dense_engine.DenseHeads.from_modules(defs, 16, args.heads_ckpt, args.refiner_ckpt, grids, args.crop).cuda()
    rows = {"occ": [], "seg": []}
    with torch.no_grad():
        for c in caps:
            occ, seg = model(c["bev"].cuda().float(), c["vol"].cuda().float())
            c["ref_occ"], c["ref_seg"] = occ.half().cpu(), seg.half().cpu()
            rows["occ"].append(err(occ, c["occ"].cuda(), -1))
            rows["seg"].append(err(seg, c["seg"].cuda(), 1))
    rep = {"frames": len(caps), "fp32_vs_deployed": {k: worst(v) for k, v in rows.items()},
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    print("fp32 DenseHeads vs deployed (worst over frames):", json.dumps(rep), flush=True)
    torch.save(caps, args.output / "captures.pt")
    c = caps[0]
    with torch.no_grad():
        torch.onnx.export(model, (c["bev"].cuda().float(), c["vol"].cuda().float()), str(args.output / "dense_heads.onnx"),
                          input_names=["bev", "vol"], output_names=["occ", "seg"], opset_version=17, dynamo=False)
    rep["onnx_MB"] = round((args.output / "dense_heads.onnx").stat().st_size / 2**20, 1)
    (args.output / "dense_export.json").write_text(json.dumps(rep, indent=1))
    print(f"export ok -> {args.output / 'dense_heads.onnx'} ({rep['onnx_MB']} MB)", flush=True)


def layer_split(eng, b, v, runs: int = 20) -> dict:
    """TensorRT per-layer times (IProfiler) summed by the DenseHeads part the layer's ONNX name starts with."""
    import tensorrt as trt

    tot = {}

    class Prof(trt.IProfiler):
        def report_layer_time(self, name, ms):
            part = next((k for k in ("occ_proj", "occ_fuse", "refiner", "occ_pred", "seg_decoder") if k in name), "other")
            if part == "other" and "MatMul" in name:
                part = "crops (MatMul)"
            tot[part] = tot.get(part, 0.0) + ms
    eng.ctx.profiler = Prof()
    for _ in range(runs):
        eng(b, v)
        torch.cuda.synchronize()
    eng.ctx.profiler = None
    return {k: round(x / runs, 2) for k, x in sorted(tot.items(), key=lambda kv: -kv[1])}


def phase_check(args):
    caps = torch.load(args.output / "captures.pt", weights_only=False)
    eng = dense_engine.DenseEngine(args.output / "dense_heads_fp16.engine")
    rows = {k: [] for k in ("occ_vs_fp32", "seg_vs_fp32", "occ_vs_deployed", "seg_vs_deployed",
                            "fp32_occ_vs_deployed", "fp32_seg_vs_deployed")}
    with torch.no_grad():
        for c in caps:
            occ, seg = eng(c["bev"].cuda(), c["vol"].cuda())
            ro, rs, do, ds = (c[k].cuda() for k in ("ref_occ", "ref_seg", "occ", "seg"))
            rows["occ_vs_fp32"].append(err(occ, ro, -1))
            rows["seg_vs_fp32"].append(err(seg, rs, 1))
            rows["occ_vs_deployed"].append(err(occ, do, -1))
            rows["seg_vs_deployed"].append(err(seg, ds, 1))
            rows["fp32_occ_vs_deployed"].append(err(ro, do, -1))
            rows["fp32_seg_vs_deployed"].append(err(rs, ds, 1))
        b, v = caps[0]["bev"].cuda(), caps[0]["vol"].cuda()
        ms = median_ms(lambda: eng(b, v), n=50)
        layers = layer_split(eng, b, v)
    rep = {"frames": len(caps), "scratch_MiB": eng.scratch_MiB, "worst": {k: worst(r) for k, r in rows.items()},
           "engine_ms": ms, "layer_ms_by_part": layers, "per_frame": rows}
    (args.output / "dense_check.json").write_text(json.dumps(rep, indent=1))
    print("engine check (worst over frames):", json.dumps(rep["worst"], indent=1), flush=True)
    print("engine latency (median, p90 ms):", ms, flush=True)
    print("engine layer time by part (ms, mean of 20 runs):", json.dumps(layers), flush=True)


def phase_identity(args):
    from e2e_latency import compare
    from qdistill import trt_student

    processor = load_processor(args.weights)
    tokens = split_tokens(args.splits, "val_quick")[::48][: args.frames_id]
    eng = args.output / "dense_heads_fp16.engine"
    outs = []
    for spec in (args.chain, f"{args.chain},dense_trt:{eng}"):
        head = load_head(args.weights)
        args_c = copy.copy(args)
        args_c.ablate = spec
        trt_head = chain(args_c, head)
        run = trt_student.load(args.student_engine, head)
        out = {}
        for t in tokens:
            metas = processor(PerceptionFrame(args.packed / t))[1]
            images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
            with torch.no_grad():
                vit, llm = run(images)
                o = run_head(head, vit, llm, copy.deepcopy(metas))
            out[t] = {"occ": o["occ"], "map": o["map"], "det": [o["boxes"], o["scores"], o["labels"]]}
        outs.append(out)
        del head, trt_head, run
        gc.collect()
        torch.cuda.empty_cache()
    rep = {"frames": len(tokens), "dense_trt_vs_chain": compare(outs[0], outs[1])}
    (args.output / "dense_identity.json").write_text(json.dumps(rep, indent=1))
    print("identity:", json.dumps(rep), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("export", "build", "check", "identity"), required=True)
    p.add_argument("--chain", default="", help="head_chain spec of the chain (logs/queue_e114.sh $ABL)")
    p.add_argument("--heads-ckpt", type=Path, default=ROOT / "weights/fine_tune.pt")
    p.add_argument("--refiner-ckpt", type=Path, default=ROOT / "weights/occ_refiner.pt")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/engines")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--crop", choices=("grid", "lerp"), default="grid", help="x / y of the occ / map crops (dense_engine)")
    p.add_argument("--frames-id", type=int, default=25)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--student-engine", type=Path, default=ROOT / "weights/engines/student_front_int8.engine")
    p.add_argument("--voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "build":
        r = build(args.output / "dense_heads.onnx", args.output / "dense_heads_fp16.engine", ln_fp32=True)
        (args.output / "dense_build.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r), flush=True)
        assert "engine" in r, r
        return
    {"export": phase_export, "check": phase_check, "identity": phase_identity}[args.phase](args)


if __name__ == "__main__":
    main()
