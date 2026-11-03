"""Export, build, measure and profile the student engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qdistill.student_io import load_student
from qdistill.data import load_frame_images, split_tokens
from qdistill.teacher import load_head
from qdistill.trt_student import (IMAGES, OUT_NAMES, FusedStudentFold, FusedStudentFPN, TrtStudentFPN)

ROOT = Path(__file__).resolve().parents[2]


def reference(student, norm, adaptor, images):
    """The deployed PyTorch path: (vit tap, adaptor levels) as the head sees them (bf16)."""
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])
        return (vit.to(torch.bfloat16), *adaptor(llm.to(torch.bfloat16).permute(0, 3, 1, 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("export", "build", "measure", "profile"), required=True)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/engines")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--workspace-gb", type=int, default=3)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--fpn-engine", type=Path, default=ROOT / "weights/engines/student_front_int8.engine",
                   help="measure: reference engine")
    p.add_argument("--fold", action="store_true", help="FusedStudentFold (adaptor folded onto the LLM tokens)")
    p.add_argument("--ln", choices=("nhwc", "nchw"), default="nhwc", help="--fold: LayerNorm2d export form")
    p.add_argument("--profile-engine", type=Path, action="append", default=[], help="--phase profile: engines to split")
    p.add_argument("--front", action="store_true", help="export FusedStudentFront: neck folded, depth_net inside")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "profile":
        return profile(args)
    if args.front:
        return front_export(args)
    stem = ("student_fold" + ("_c" if args.ln == "nchw" else "")) if args.fold else "student_fpn"
    onnx_path = args.output / f"{stem}.onnx"
    eng = args.output / f"{stem}_fp16.engine"
    tokens = split_tokens(args.splits, "val_quick")[::48][: args.frames]

    if args.phase == "export":
        student, norm = load_student(args.ckpt, args.cache)
        head = load_head(args.weights)
        adaptor = head.bev_modeling.adaptor
        w = (FusedStudentFold(student, norm, adaptor, args.ln) if args.fold else FusedStudentFPN(student, norm, adaptor)).cuda().float().eval()
        ref_adaptor = __import__("copy").deepcopy(adaptor).float()
        del head
        torch.cuda.empty_cache()
        torch.backends.mha.set_fastpath_enabled(False)  # nn.TransformerEncoder fast path has no ONNX export
        x = load_frame_images(args.packed, tokens[0], "jpeg_gpu")[None].cuda().float()
        with torch.no_grad():
            o = w(x)
            vit_n, llm_n = student.float()(x)
            r = (norm.denormalize("vit", vit_n[0]), *ref_adaptor(norm.denormalize("llm", llm_n[0]).permute(0, 3, 1, 2)))
            diff = [float((a - b).abs().max() / b.abs().max()) for a, b in zip(o, r)]  # LayerNorm reorder: fp32 rounding only
        print("fused wrapper vs modules (fp32), max abs diff / max |ref| per output:", dict(zip(OUT_NAMES, diff)), flush=True)
        assert all(d < 1e-3 for d in diff), diff  # TF32 convs amplify the reorder (l0 1.5e-4)
        torch.onnx.export(w, (x,), str(onnx_path), input_names=["images"], output_names=list(OUT_NAMES) + (["z"] if args.fold else []),
                          opset_version=17, dynamo=False)
        print(f"export ok ({onnx_path.stat().st_size / 2**20:.0f} MB) -> {onnx_path}", flush=True)
    elif args.phase == "build":
        from qdistill.trt_utils import build_engine
        r = build_engine(onnx_path, "images", IMAGES, eng, precision="fp16", workspace_gb=args.workspace_gb)
        print(json.dumps(r, default=str), flush=True)
        assert not r.get("failed"), r
    else:
        from qdistill.trt_utils import median_ms
        student, norm = load_student(args.ckpt, args.cache)
        head = load_head(args.weights)
        adaptor = head.bev_modeling.adaptor  # the PyTorch module, kept before the stand-in replaces it
        run = TrtStudentFPN(eng, head)
        fpn = TrtStudentFPN(args.fpn_engine, head) if args.fold else None
        vs_fpn = {n: [] for n in OUT_NAMES}
        acc = {n: {"nmse": [], "cos": [], "max_abs": [], "nonfinite": 0} for n in OUT_NAMES}
        x = None
        for t in tokens:
            x = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
            ref = reference(student, norm, adaptor, x)
            with torch.no_grad():
                vit, _ = run(x)
            got = (vit, *run.levels)
            if fpn is not None:
                with torch.no_grad():
                    fv, _ = fpn(x)
                for n, a, b in zip(OUT_NAMES, got, (fv, *fpn.levels)):
                    a, b = a.float(), b.float()
                    vs_fpn[n].append(float(((a - b) ** 2).mean() / b.var()))
            for n, a, b in zip(OUT_NAMES, got, ref):
                a, b = a.float(), b.float()
                acc[n]["nonfinite"] += int((~torch.isfinite(a)).sum())
                acc[n]["nmse"].append(float(((a - b) ** 2).mean() / b.var()))
                ch = 1 if n != "vit" else -1  # channel dim: levels NCHW, vit tap N H W C
                acc[n]["cos"].append(float(torch.nn.functional.cosine_similarity(a, b, dim=ch).mean()))
                acc[n]["max_abs"].append(float((a - b).abs().max()))
        res = {"frames": len(tokens), "engine": str(eng), **run.info(), "agreement": {
            n: {"nmse_mean": float(np.mean(v["nmse"])), "nmse_max": float(np.max(v["nmse"])),
                "cos_mean": float(np.mean(v["cos"])), "max_abs": float(np.max(v["max_abs"])), "nonfinite": v["nonfinite"]}
            for n, v in acc.items()}}
        x16 = x.to(torch.float16).contiguous()
        t_pt = median_ms(lambda: reference(student, norm, adaptor, x), n=50)
        t_trt = median_ms(lambda: run(x16), n=50)
        res["latency_ms"] = {"pytorch_bf16": t_pt, "trt_fp16": t_trt}
        if fpn is not None:
            res["latency_ms"]["fpn_engine"] = median_ms(lambda: fpn(x16), n=50)
            res["vs_fpn_engine_nmse"] = {n: {"mean": float(np.mean(v)), "max": float(np.max(v))} for n, v in vs_fpn.items()}
        name = f"{stem}_measure.json" if args.fold else "trt_student_fpn_measure.json"
        (args.output / name).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1), flush=True)


def front_export(args):
    """FusedStudentFront ONNX, checked in fp32 against vit_neck + depth_net on the de-normalised tap."""
    from qdistill import trt_modules
    from qdistill.trt_student import FusedStudentFront

    assert args.phase == "export", args.phase
    student, norm = load_student(args.ckpt, args.cache)
    head = load_head(args.weights)
    bev = head.bev_modeling
    with torch.no_grad():
        neck_ref, depth_ref = trt_modules.fp32_copy(bev.vit_neck).cuda(), trt_modules.fp32_copy(bev.depth_net).cuda()
        w = FusedStudentFront(student, norm, bev.adaptor, bev.vit_neck.float(), trt_modules.fp32_copy(bev.depth_net)).cuda().float().eval()
    del head
    torch.cuda.empty_cache()
    torch.backends.mha.set_fastpath_enabled(False)
    x = load_frame_images(args.packed, split_tokens(args.splits, "val_quick")[0], "jpeg_gpu")[None].cuda().float()
    with torch.no_grad():
        o = w(x)
        vit_n, _ = student(x)
        n = neck_ref(norm.denormalize("vit", vit_n[0]).permute(0, 3, 1, 2))[0]
        diff = [float((a - b).abs().max() / b.abs().max()) for a, b in ((o[0], n), (o[1], depth_ref(n)))]
    print("front vs vit_neck + depth_net (fp32), max abs diff / max |ref|:", diff, flush=True)
    assert all(d < 1e-3 for d in diff), diff
    path = args.output / "student_front.onnx"
    torch.onnx.export(w, (x,), str(path), input_names=["images"], output_names=["neck", "depth", *OUT_NAMES[1:], "z"],
                      opset_version=17, dynamo=False)
    print(f"export ok ({path.stat().st_size / 2**20:.0f} MB) -> {path}", flush=True)








PARTS = (("llm_out", "llm_out"), ("adaptor/stages.0", "adaptor L0 (x4)"), ("adaptor/stages.1", "adaptor L1 (x2)"),
         ("adaptor/stages.2", "adaptor L2 (x1)"), ("adaptor/stages.3", "adaptor L3 (x0.5)"),
         ("fold/l0", "adaptor L0 (x4)"), ("fold/l1", "adaptor L1 (x2)"), ("fold/l2", "adaptor L2 (x1)"),
         ("fold/l3", "adaptor L3 (x0.5)"), ("student/encoder", "LLM transformer"), ("llm_in", "LLM transformer"),
         ("pos_embed", "LLM transformer"), ("cam_embed", "LLM transformer"), ("_gemm_mha", "LLM transformer"),
         ("student/backbone", "backbone"), ("lat16", "FPN laterals"), ("lat32", "FPN laterals"),
         ("vit_neck", "ViT path"), ("vit_out", "ViT path"), ("Reformatting", "reformat copies"))


def profile(args):
    """TensorRT per-layer times of student engines, summed by part (first matching key of the layer name)."""
    import re

    import tensorrt as trt

    x = load_frame_images(args.packed, split_tokens(args.splits, "val_quick")[0], "jpeg_gpu")[None].cuda()
    x16 = x.to(torch.float16).contiguous()
    res = {}
    for path in args.profile_engine:
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        eng = rt.deserialize_cuda_engine(Path(path).read_bytes())
        ctx = eng.create_execution_context()
        bufs = []
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                b = x16 if n == "images" else torch.randn(tuple(eng.get_tensor_shape(n)), device="cuda").half()
                ctx.set_input_shape(n, tuple(b.shape))
                ctx.set_tensor_address(n, b.data_ptr())
                bufs.append(b)
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                b = torch.empty(tuple(eng.get_tensor_shape(n)), dtype=torch.float16, device="cuda")
                bufs.append(b)
                ctx.set_tensor_address(n, b.data_ptr())
        tot, layers = {}, {}

        class Prof(trt.IProfiler):
            def report_layer_time(self, name, ms):
                part = next((v for k, v in PARTS if k in name), None)
                if part is None:
                    m = re.search(r"myl(\d+)", name)
                    part = f"Myelin glue (region {m.group(1)})" if m else "other"
                tot[part] = tot.get(part, 0.0) + ms
                layers[name] = layers.get(name, 0.0) + ms
        stream = torch.cuda.current_stream().cuda_stream
        for _ in range(5):
            ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        ctx.profiler = Prof()
        runs = 20
        for _ in range(runs):
            ctx.execute_async_v3(stream)
            torch.cuda.synchronize()
        top = sorted(layers.items(), key=lambda kv: -kv[1])[:40]
        res[str(path)] = {"sum_ms": round(sum(tot.values()) / runs, 2),
                          "by_part_ms": {k: round(v / runs, 2) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])},
                          "top_layers_ms": {k[:160]: round(v / runs, 3) for k, v in top}}
        print(path, json.dumps(res[str(path)]["by_part_ms"]), "sum", res[str(path)]["sum_ms"], flush=True)
        del ctx, eng
    (args.output / "student_profile.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
