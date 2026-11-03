"""Export and build voxel block TensorRT engines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qdistill.trt_utils import act_amax, build_engine, conv_input_stats, insert_qdq

ROOT = Path(__file__).resolve().parents[2]
X, Y, Z, C = 200, 200, 16, 256  # upstream voxel_shape and embed_dim (checked against the head at measure time)


class FrozenStack(nn.Module):
    """[1, 1, X, Y, Z, C] (kernel layout) -> [1, C, Z, Y, X]: exactly teacher_encoding, BN in eval mode."""

    def __init__(self, conv_layer: nn.ModuleList):
        super().__init__()
        self.conv_layer = conv_layer

    def forward(self, s):
        x = s.sum(1).permute(0, 4, 3, 2, 1)
        for layer in self.conv_layer:
            x = layer(x)
        return x


def load_stack(args, device="cpu"):
    """Upstream view_trans conv stack with the teacher's weights, without loading the VLM."""
    from qdistill.teacher import load_head

    head = load_head(args.weights, device=device)
    vt = head.bev_modeling.view_trans
    assert list(vt.voxel_shape) == [X, Y, Z], vt.voxel_shape
    return head, FrozenStack(vt.conv_layer).eval()


def target_models(args):
    """{name: loader() -> fp32 CPU module taking s [1, 1, X, Y, Z, C]}: the exact stack + the anchor blocks."""
    import copy

    from qdistill.light_blocks import load_light_voxel

    def stack():
        head, st = load_stack(args)
        st = copy.deepcopy(st).float().eval()
        del head
        return st

    out = {"voxel_stack": stack}
    for b in args.anchor_blocks:
        out[Path(b).parent.name] = lambda b=b: load_light_voxel(b, device="cpu").float().eval()
    return out


def phase_export(args):
    res = {}
    for name, load in target_models(args).items():
        path = args.output / f"{name}.onnx"
        if path.exists():
            res[name] = {"onnx": str(path), "existed": True}
            continue
        model = load()
        for m in model.modules():  # ReLU(inplace) is fine for tracing; keep the graph plain
            if isinstance(m, nn.ReLU):
                m.inplace = False
        dummy = torch.randn(1, 1, 8, 8, 4, C)  # tiny trace input; engine profiles pin the real shape
        torch.onnx.export(model, (dummy,), str(path), input_names=["s"], output_names=["y"], opset_version=17,
                          dynamic_axes={"s": {2: "X", 3: "Y", 4: "Z"}, "y": {2: "Z", 3: "Y", 4: "X"}}, dynamo=False)
        import onnx

        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        ops = sorted({n.op_type for n in m.graph.node})
        with torch.no_grad():  # ONNX graph == PyTorch on the tiny input (onnx reference evaluator, CPU; small here)
            from onnx.reference import ReferenceEvaluator

            ref = model(dummy).numpy()
            got = ReferenceEvaluator(m).run(None, {"s": dummy.numpy()})[0]
        err = float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12))
        print(f"export {name} -> {path}: ops {ops}, max rel err vs torch {err:.2e}", flush=True)
        assert err < 1e-4, err
        res[name] = {"onnx": str(path), "ops": ops, "max_rel_err": err}
    return res


def phase_qdq(args):
    """Conv-input ranges of each target on all calibration volumes (one GPU pass, fp32), then one Q/DQ ONNX per
    (method, n); the n-subsets are the nested calib_subset ones."""
    files = sorted(args.calib_dir.glob("[0-9][0-9][0-9].npy"))
    pos = {f: i for i, f in enumerate(files)}
    res = {}
    for name, load in target_models(args).items():
        model = load().cuda()
        stats = conv_input_stats(model, (torch.from_numpy(np.load(f)).cuda().float() for f in files))
        del model
        torch.cuda.empty_cache()
        (args.output / f"{name}_conv_stats.json").write_text(json.dumps(stats))
        shapes = [c["w"] for c in stats[0]]
        res[name] = {"convs": len(shapes), "volumes": len(files)}
        for n in args.calib_n:
            idx = [pos[f] for f in calib_subset(files, n)]
            for method in args.qdq_methods:
                amax = act_amax(stats, idx, method)
                out = args.output / f"{name}_q{method}_c{n}.onnx"
                insert_qdq(args.output / f"{name}.onnx", out, amax, shapes)
                res[name][f"q{method}_c{n}"] = [round(a, 3) for a in amax]
        for n in args.fp8_calib_n:  # FP8 E4M3: max range; per-layer simulation 0.0005 raw nMSE
            idx = [pos[f] for f in calib_subset(files, n)]
            insert_qdq(args.output / f"{name}.onnx", args.output / f"{name}_fp8max_c{n}.onnx", act_amax(stats, idx, "max"),
                       shapes, fmt="fp8")
        print(f"qdq {name}: {res[name]}", flush=True)
    return res


def student_stack(args):
    """Frozen 4k student + head on the GPU, as distill_voxel_block.py builds them."""
    from qdistill.student import TapNormalizer, TapStudent
    from qdistill.teacher import load_head

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    a = ck["args"]
    stats = a.get("tap_stats") or args.cache / f"tap_stats_{a['train_split'].replace(':', '_')}.npz"
    student = TapStudent(a["backbone"], pretrained=False).cuda().eval()
    student.load_state_dict(ck["model"])
    norm = TapNormalizer(dict(np.load(stats))).cuda()
    head = load_head(args.weights)
    for q in list(student.parameters()) + list(head.parameters()):
        q.requires_grad_(False)
    return student, norm, head


def frame_tensors(student, norm, head, packed: Path, tok: str):
    from distill_voxel_block import step_tensors
    from qdistill.data import IMAGE_SIZE, load_images
    from qwen_drive_perception.dataset import PerceptionFrame

    fd = packed / tok
    return step_tensors(student, norm, head, load_images(fd), PerceptionFrame(fd).img_metas(image_size=IMAGE_SIZE))


def calib_subset(files, n):
    """n evenly spaced volumes of the dumped set (12 / 24 / 32 of 32 are nested: 12 c 24 c 32)."""
    idx = sorted({int(round(i * len(files) / n)) for i in range(n)})
    assert len(idx) == n and idx[-1] < len(files), (n, len(files))
    return [files[i] for i in idx]


def phase_calib(args):
    from qdistill.data import split_tokens

    train = split_tokens(args.splits, "s1_C4k")
    toks = train[:: max(1, len(train) // args.calib_frames)][: args.calib_frames]
    d = args.calib_dir
    d.mkdir(parents=True, exist_ok=True)
    todo = [(i, t) for i, t in enumerate(toks) if not (d / f"{i:03d}.npy").exists()]  # resumable
    if todo:
        student, norm, head = student_stack(args)
        for i, tok in todo:
            s, _ = frame_tensors(student, norm, head, args.packed, tok)
            tmp = d / f"{i:03d}.tmp.npy"
            np.save(tmp, s.to(torch.float16).cpu().numpy())
            tmp.rename(d / f"{i:03d}.npy")
            print(f"calib {i + 1}/{len(toks)} {tok}", flush=True)
            del s
            torch.cuda.empty_cache()  # frame k+1 otherwise OOMs in the voxel-pool kernel (03:59 / 04:10 runs)
    (d / "frames.json").write_text(json.dumps(toks))
    return {"frames": toks, "dir": str(d), "dumped_now": len(todo)}


def engine_names(args):
    return (["fp16"] + [f"int8q_{m}_c{n}" for m in args.qdq_methods for n in args.calib_n]
            + [f"fp8q_max_c{n}" for n in args.fp8_calib_n])


def onnx_of(args, tgt, name):
    if name == "fp16":
        return args.output / f"{tgt}.onnx", "fp16"
    if name.startswith("fp8q_"):
        return args.output / f"{tgt}_fp8{name[len('fp8q_'):]}.onnx", "fp8q"
    return args.output / f"{tgt}_q{name[len('int8q_'):]}.onnx", "int8q"


def phase_build(args):
    res = {}
    for tgt in target_models(args):
        for name in engine_names(args):
            if args.only and name not in args.only:
                continue
            onnx_path, prec = onnx_of(args, tgt, name)
            path = args.output / f"{tgt}_{name}.engine"
            if path.exists():
                res[f"{tgt}_{name}"] = {"engine": str(path), "existed": True}
                continue
            r = build_engine(onnx_path, "s", (1, 1, X, Y, Z, C), path, prec,
                             workspace_gb=args.workspace_gb)
            res[f"{tgt}_{name}"] = r
            print(f"build {tgt}_{name}: {r}", flush=True)
    if not any("engine" in r for r in res.values()):
        raise SystemExit("no engine built")
    return res


class TrtStack:
    """Runs a serialized engine on torch tensors (fp16 in kernel layout -> fp16 [1, C, Z, Y, X])."""

    def __init__(self, path: Path):
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.ctx.set_input_shape("s", (1, 1, X, Y, Z, C))
        self.out = torch.empty((1, C, Z, Y, X), dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address("y", self.out.data_ptr())

    def __call__(self, s16: torch.Tensor) -> torch.Tensor:
        assert s16.dtype == torch.float16 and s16.is_contiguous()
        self.ctx.set_tensor_address("s", s16.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def median_ms(fn, n=100, warm=10):
    ms = []
    for k in range(warm + n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if k >= warm:
            ms.append(1000 * (time.perf_counter() - t))
    return float(np.median(ms)), float(np.percentile(ms, 90))


def phase_measure(args):
    """One variant at a time (engines + student / head do not all fit 8 GB); s, t recomputed per pass (no-grad,
    deterministic up to cuDNN noise)."""
    from distill_voxel_block import losses
    from qdistill.data import split_tokens
    from qdistill.light_blocks import load_light_voxel, teacher_encoding

    student, norm, head = student_stack(args)
    vt = head.bev_modeling.view_trans
    assert list(vt.voxel_shape) == [X, Y, Z], vt.voxel_shape
    variants = [(f"trt_{e.stem}", e) for tgt in target_models(args) for e in sorted(args.output.glob(f"{tgt}_*.engine"))]
    variants = [(v, e) for v, e in variants if e.exists() and (not args.only or any(v.endswith("_" + o) for o in args.only))]
    variants += [(Path(b).parent.name, Path(b)) for b in args.anchor_blocks]
    val = split_tokens(args.splits, "val_quick")[:: args.val_stride][: args.max_frames or None]
    print(f"variants {[v for v, _ in variants]}; {len(val)} val frames", flush=True)

    accuracy, lat = {}, {}
    with torch.no_grad():
        s, _ = frame_tensors(student, norm, head, args.packed, val[0])

        def pt_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_encoding(vt, s)
        lat["pytorch_bf16_permute_conv"] = median_ms(pt_bf16)
        lat["cast_bf16_to_fp16"] = median_ms(lambda: s.to(torch.float16).contiguous())
        del s
    for name, path in variants:
        if name.startswith("trt_"):
            m = TrtStack(path)
            fn = lambda s: m(s.to(torch.float16).contiguous())
        else:
            m = load_light_voxel(path)

            def fn(s):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return m(s)
        acc = {}
        with torch.no_grad():
            for i, tok in enumerate(val):
                s, t = frame_tensors(student, norm, head, args.packed, tok)
                _, parts = losses(head, fn(s), t, 0.1)
                for k in ("bev_q", "occ_p", "raw"):
                    acc.setdefault(k, []).append(float(parts[k]))
                if i == 0:
                    s16 = s.to(torch.float16).contiguous()
                    lat[name] = median_ms((lambda: m(s16)) if name.startswith("trt_") else (lambda: fn(s)))
                    del s16
                del s, t
        accuracy[name] = {k: float(np.mean(v)) for k, v in acc.items()}
        print(f"{name:28s} " + "  ".join(f"{k} {v:.4f}" for k, v in accuracy[name].items())
              + f"  {lat[name][0]:.1f} ms (p90 {lat[name][1]:.1f})", flush=True)
        del m, fn
        torch.cuda.empty_cache()
    lat = {k: {"median_ms": round(v[0], 1), "p90_ms": round(v[1], 1)} for k, v in lat.items()}
    print(f"pytorch bf16 permute+conv {lat['pytorch_bf16_permute_conv']['median_ms']} ms; "
          f"bf16->fp16 cast {lat['cast_bf16_to_fp16']['median_ms']} ms", flush=True)
    return {"val_frames": len(val), "accuracy": accuracy, "latency": lat,
            "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["export", "calib", "qdq", "build", "measure"], required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/voxel_block")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--student", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--qdq-methods", nargs="+", default=["mse", "pct", "max"], choices=["mse", "pct", "max"],
                   help="explicit-INT8 activation ranges: MSE-optimal clip, 99.99th percentile or max of |conv input|")
    p.add_argument("--fp8-calib-n", type=int, nargs="*", default=[32], help="FP8 Q/DQ (max range) from these subset sizes")
    p.add_argument("--only", nargs="*", default=[], help="build / measure: only these engine names (e.g. fp16)")
    p.add_argument("--calib-frames", type=int, default=32, help="s1_C4k volumes dumped for INT8 calibration (327 MB each)")
    p.add_argument("--calib-dir", type=Path, default=ROOT / "cache/block_calib")
    p.add_argument("--calib-n", type=int, nargs="+", default=[12, 24, 32], help="int8 engines from these subset sizes")
    p.add_argument("--workspace-gb", type=int, default=2)
    p.add_argument("--val-stride", type=int, default=12, help="as distill_voxel_block.py: 101 val_quick frames")
    p.add_argument("--max-frames", type=int, default=0, help="smoke test: limit the accuracy frames")
    p.add_argument("--anchor-blocks", nargs="*", default=[str(ROOT / "runs/block_r64/best.pt")])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    res = {"export": phase_export, "calib": phase_calib, "qdq": phase_qdq, "build": phase_build,
           "measure": phase_measure}[args.phase](args)
    out = args.output / f"trt_voxel_{args.phase}{'_smoke' if args.max_frames else ''}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
