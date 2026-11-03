"""Export and build occupancy refiner TensorRT engines."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qdistill.trt_utils import TrtRunner, act_amax, build_engine, calib_subset, conv_input_stats, insert_qdq, median_ms

ROOT = Path(__file__).resolve().parents[2]
IN_SHAPE = (1, 16, 16, 200, 200)    # [B, C, Z, Y, X]; checked against the live head at calib / measure
OUT_SHAPE = (1, 32, 16, 200, 200)
EMPTY = 9                           # OCC_EMPTY_LABEL


def refiner_of(head):
    return head.bev_modeling.head.transformer.occ_decoder


def phase_export(args):
    from qdistill.teacher import load_head

    head = load_head(args.weights, device="cpu")
    ref = copy.deepcopy(refiner_of(head)).float().eval()
    del head
    for m in ref.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    dummy = torch.randn(1, 16, 4, 24, 24)  # tiny trace input (3 x stride-2 in Y / X); engines pin the real shape
    path = args.output / "occ_refiner.onnx"
    torch.onnx.export(ref, (dummy,), str(path), input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x": {2: "Z", 3: "Y", 4: "X"}, "y": {2: "Z", 3: "Y", 4: "X"}}, dynamo=False)
    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    n_params = sum(p.numel() for p in ref.parameters())
    err = None
    print(f"export -> {path}: {n_params / 1e6:.2f}M params, ops {ops}, max rel err vs torch {err}", flush=True)
    return {"onnx": str(path), "ops": ops, "max_rel_err": err, "params_M": n_params / 1e6}


class Captured:
    """Runs the full head on the student's taps and keeps the refiner's input / output (bf16 on the GPU)."""

    def __init__(self, args):
        from qdistill.student_io import load_student
        from qdistill.teacher import load_head

        self.args = args
        self.student, self.norm = load_student(args.student, args.cache)
        self.head = load_head(args.weights)
        for q in list(self.student.parameters()) + list(self.head.parameters()):
            q.requires_grad_(False)
        self.ref = refiner_of(self.head)
        self.io = {}
        self.ref.register_forward_hook(lambda m, a, out: self.io.update(x=a[0].detach(), y=out.detach()))

    def __call__(self, tok):
        from qdistill.data import IMAGE_SIZE, load_frame_images
        from qdistill.teacher import run_head
        from qwen_drive_perception.dataset import PerceptionFrame

        images = load_frame_images(self.args.packed, tok, "jpeg")[None].cuda()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vit_n, llm_n = self.student(images)
            vit, llm = self.norm.denormalize("vit", vit_n[0]), self.norm.denormalize("llm", llm_n[0])
            run_head(self.head, vit, llm, PerceptionFrame(self.args.packed / tok).img_metas(image_size=IMAGE_SIZE))
        x, y = self.io.pop("x"), self.io.pop("y")
        assert tuple(x.shape) == IN_SHAPE and tuple(y.shape) == OUT_SHAPE, (x.shape, y.shape)
        return x, y

    def labels(self, y):
        """Occupancy labels as the head decodes them: occ_pred_head on [B, X, Y, Z, 32], argmax."""
        head = self.head.bev_modeling.head.transformer.occ_pred_head
        return head(y.to(torch.bfloat16).permute(0, 4, 3, 2, 1)).argmax(-1)


def phase_calib(args):
    from qdistill.data import split_tokens

    train = split_tokens(args.splits, "s1_C4k")
    toks = train[:: max(1, len(train) // args.calib_frames)][: args.calib_frames]
    d = args.calib_dir
    d.mkdir(parents=True, exist_ok=True)
    todo = [(i, t) for i, t in enumerate(toks) if not (d / f"{i:03d}.npy").exists()]  # resumable
    if todo:
        cap = Captured(args)
        for i, tok in todo:
            x, _ = cap(tok)
            tmp = d / f"{i:03d}.tmp.npy"
            np.save(tmp, x.to(torch.float16).cpu().numpy())
            tmp.rename(d / f"{i:03d}.npy")
            print(f"calib {i + 1}/{len(toks)} {tok}", flush=True)
            del x
            torch.cuda.empty_cache()  # frame k+1 otherwise OOMs in the voxel-pool kernel (03:59 / 04:10 runs)
    (d / "frames.json").write_text(json.dumps(toks))
    return {"frames": toks, "dir": str(d), "dumped_now": len(todo)}


def load_refiner(args, device="cuda"):
    """(refiner, occ_pred_head) from the head on the CPU; only these two go to ``device``."""
    from qdistill.teacher import load_head

    head = load_head(args.weights, device="cpu")
    tr = head.bev_modeling.head.transformer
    ref, pred = tr.occ_decoder, tr.occ_pred_head
    del head, tr
    return ref.to(device).eval(), pred.to(device).eval()


def phase_qdq(args):
    files = sorted(args.calib_dir.glob("[0-9][0-9][0-9].npy"))
    pos = {f: i for i, f in enumerate(files)}
    ref, _ = load_refiner(args, device="cpu")
    model = copy.deepcopy(ref).float().cuda().eval()
    del ref
    stats = conv_input_stats(model, (torch.from_numpy(np.load(f)).cuda().float().reshape(IN_SHAPE) for f in files))
    del model
    torch.cuda.empty_cache()
    (args.output / "occ_refiner_conv_stats.json").write_text(json.dumps(stats))
    shapes = [c["w"] for c in stats[0]]
    res = {"convs": len(shapes), "volumes": len(files)}
    for n in args.calib_n:
        idx = [pos[f] for f in calib_subset(files, n)]
        for method in args.qdq_methods:
            amax = act_amax(stats, idx, method)
            insert_qdq(args.output / "occ_refiner.onnx", args.output / f"occ_refiner_q{method}_c{n}.onnx", amax, shapes)
            res[f"q{method}_c{n}"] = [round(a, 3) for a in amax]
    for n in args.fp8_calib_n:  # FP8 E4M3, max range (INT8 activations too coarse on these volumes)
        idx = [pos[f] for f in calib_subset(files, n)]
        insert_qdq(args.output / "occ_refiner.onnx", args.output / f"occ_refiner_fp8max_c{n}.onnx", act_amax(stats, idx, "max"),
                   shapes, fmt="fp8")
    print(f"qdq occ_refiner: {res}", flush=True)
    return res


def engine_names(args):
    return (["fp16"] + [f"int8q_{m}_c{n}" for m in args.qdq_methods for n in args.calib_n]
            + [f"fp8q_max_c{n}" for n in args.fp8_calib_n])


def phase_build(args):
    res = {}
    for name in engine_names(args):
        if args.only and name not in args.only:
            continue
        path = args.output / f"occ_refiner_{name}.engine"
        if path.exists():
            res[name] = {"engine": str(path), "existed": True}
            continue
        if name == "fp16":
            onnx_path, prec = args.output / "occ_refiner.onnx", "fp16"
        elif name.startswith("fp8q_"):
            onnx_path, prec = args.output / f"occ_refiner_fp8{name[len('fp8q_'):]}.onnx", "fp8q"
        else:
            onnx_path, prec = args.output / f"occ_refiner_q{name[len('int8q_'):]}.onnx", "int8q"
        r = build_engine(onnx_path, "x", IN_SHAPE, path, prec, workspace_gb=args.workspace_gb)
        res[name] = r
        print(f"build {name}: {r}", flush=True)
    if not any("engine" in r for r in res.values()):
        raise SystemExit("no engine built")
    return res


def agreement(lt, lp):
    """Label agreement of TRT (lp) vs PyTorch (lt): all voxels, PyTorch-occupied voxels, mIoU over non-empty classes."""
    occ = lt != EMPTY
    ious = []
    for c in range(EMPTY):
        a, b = lt == c, lp == c
        u = (a | b).sum().item()
        if u:
            ious.append((a & b).sum().item() / u)
    return {"agree_all": (lt == lp).float().mean().item(), "agree_occupied": (lt[occ] == lp[occ]).float().mean().item(),
            "miou_vs_pytorch": float(np.mean(ious)) if ious else float("nan")}


def val_tokens(args):
    from qdistill.data import split_tokens

    return split_tokens(args.splits, "val_quick")[:: args.val_stride][: args.max_frames or None]


def phase_ref(args):
    """PyTorch refiner input / output (fp16) of the val frames with the full head on the student's taps
    (resumable). The engines are compared against these in `measure`."""
    val = val_tokens(args)
    d = args.ref_dir
    d.mkdir(parents=True, exist_ok=True)
    todo = [(i, t) for i, t in enumerate(val) if not (d / f"{i:03d}_y.npy").exists()]
    if todo:
        cap = Captured(args)
        for i, tok in todo:
            x, y = cap(tok)
            for k, v in (("x", x), ("y", y)):
                tmp = d / f"{i:03d}_{k}.tmp.npy"
                np.save(tmp, v.to(torch.float16).cpu().numpy())
                tmp.rename(d / f"{i:03d}_{k}.npy")
            del x, y
            torch.cuda.empty_cache()
            print(f"ref {i + 1}/{len(val)} {tok}", flush=True)
    (d / "frames.json").write_text(json.dumps(val))
    return {"frames": len(val), "dir": str(d), "dumped_now": len(todo)}


def phase_measure(args):
    """Refiner + occ_pred_head on the GPU only; inputs / PyTorch outputs from `ref`. The PyTorch reference is the
    bf16 refiner as the head runs it (captured y), so nmse / agreement include fp16-vs-bf16 rounding."""
    val = val_tokens(args)
    d = args.ref_dir
    assert all((d / f"{i:03d}_y.npy").exists() for i in range(len(val))), "run --phase ref first"
    load = lambda i, k: torch.from_numpy(np.load(d / f"{i:03d}_{k}.npy")).cuda()
    ref, pred = load_refiner(args)

    def labels(y):
        return pred(y.to(torch.bfloat16).permute(0, 4, 3, 2, 1)).argmax(-1)

    names = [n for n in engine_names(args) if (args.output / f"occ_refiner_{n}.engine").exists()
             and (not args.only or n in args.only)]
    print(f"engines {names}; {len(val)} val frames", flush=True)
    lat, accuracy = {}, {}
    x0 = load(0, "x").to(torch.bfloat16)
    with torch.no_grad():
        lat["pytorch_bf16"] = median_ms(lambda: ref(x0))
        lat["cast_bf16_to_fp16"] = median_ms(lambda: x0.to(torch.float16).contiguous())
    x16 = x0.to(torch.float16).contiguous()
    del x0, ref
    torch.cuda.empty_cache()
    for name in names:
        run = TrtRunner(args.output / f"occ_refiner_{name}.engine", "x", IN_SHAPE, "y", OUT_SHAPE)
        lat[f"trt_{name}"] = median_ms(lambda: run(x16))
        acc = {}
        with torch.no_grad():
            for i in range(len(val)):
                y = load(i, "y")
                yp = run(load(i, "x").contiguous())
                yf, ypf = y.float(), yp.float()
                acc.setdefault("nmse", []).append(float(((ypf - yf) ** 2).mean() / yf.var().clamp_min(1e-12)))
                for k, v in agreement(labels(y), labels(yp)).items():
                    acc.setdefault(k, []).append(v)
                del y, yp, yf, ypf
        accuracy[f"trt_{name}"] = {k: float(np.nanmean(v)) for k, v in acc.items()}
        m = lat[f"trt_{name}"]
        print(f"trt_{name:18s} " + "  ".join(f"{k} {v:.4f}" for k, v in accuracy[f"trt_{name}"].items())
              + f"  {m[0]:.1f} ms (p90 {m[1]:.1f})", flush=True)
        del run
        torch.cuda.empty_cache()
    lat = {k: {"median_ms": round(v[0], 1), "p90_ms": round(v[1], 1)} for k, v in lat.items()}
    print(f"pytorch bf16 refiner {lat['pytorch_bf16']['median_ms']} ms; bf16->fp16 cast {lat['cast_bf16_to_fp16']['median_ms']} ms",
          flush=True)
    return {"val_frames": len(val), "accuracy": accuracy, "latency": lat,
            "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["export", "calib", "qdq", "build", "ref", "measure"], required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/occ_refiner")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--student", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--qdq-methods", nargs="+", default=["mse", "pct", "max"], choices=["mse", "pct", "max"],
                   help="explicit-INT8 activation ranges: MSE-optimal clip, 99.99th percentile or max of |conv input|")
    p.add_argument("--fp8-calib-n", type=int, nargs="*", default=[32], help="FP8 Q/DQ (max range) from these subset sizes")
    p.add_argument("--only", nargs="*", default=[], help="build / measure: only these engine names (e.g. fp16)")
    p.add_argument("--ref-dir", type=Path, default=ROOT / "cache/refiner_val", help="ref phase output (~6 GB)")
    p.add_argument("--calib-frames", type=int, default=32, help="s1_C4k refiner inputs dumped (20 MB each)")
    p.add_argument("--calib-dir", type=Path, default=ROOT / "cache/refiner_calib")
    p.add_argument("--calib-n", type=int, nargs="+", default=[12, 24, 32], help="int8 engines from these subset sizes")
    p.add_argument("--workspace-gb", type=int, default=2)
    p.add_argument("--val-stride", type=int, default=12, help="as distill_voxel_block.py: 101 val_quick frames")
    p.add_argument("--max-frames", type=int, default=0, help="smoke test: limit the accuracy frames")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    res = {"export": phase_export, "calib": phase_calib, "qdq": phase_qdq, "build": phase_build, "ref": phase_ref,
           "measure": phase_measure}[args.phase](args)
    out = args.output / f"trt_occ_{args.phase}{'_smoke' if args.max_frames else ''}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
