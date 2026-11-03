"""INT8 student engine: calibrate activation ranges, share scales per block, insert Q/DQ, build, measure."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qdistill.student_io import load_student
from qdistill import student_quant as sq
from qdistill.data import load_frame_images, split_tokens
from qdistill.teacher import load_head
from qdistill.trt_student import IMAGES, OUT_NAMES, FusedStudentFold, TrtStudentFPN
from qdistill.trt_utils import act_amax, conv_input_stats, insert_qdq

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("mse", "pct", "max")


def images(args, t):
    return load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda().float()


def val_tokens(args):
    return split_tokens(args.splits, args.val_split)[:: args.val_stride][: args.val_frames]


def fold_model(args, ckpt=None):
    """FusedStudentFold fp32 on the GPU (the head is only needed for its adaptor) and the raw student inside it."""
    student, norm = load_student(ckpt or args.ckpt, args.cache)
    head = load_head(args.weights)
    with torch.no_grad():  # the folded biases are buffers: computed with grad they would keep a graph to the adaptor
        w = FusedStudentFold(student, norm, head.bev_modeling.adaptor).cuda().float().eval()
    del head
    torch.cuda.empty_cache()
    torch.backends.mha.set_fastpath_enabled(False)  # same path as the ONNX export
    return w.requires_grad_(False)


def calib(args):
    student, _ = load_student(args.ckpt, args.cache)
    tok = split_tokens(args.splits, args.train_split)
    tok = [tok[int(i * len(tok) / args.calib_frames)] for i in range(args.calib_frames)]
    t0 = time.time()

    def volumes():
        for k, t in enumerate(tok):
            if k % 50 == 0:
                print(f"{k}/{len(tok)} {time.time() - t0:.0f} s", flush=True)
            yield images(args, t).flatten(0, 1)

    stats = conv_input_stats(student.backbone, volumes(), sample=args.sample)
    names = ["backbone." + c["name"] for c in stats[0]]
    assert len(names) == len(set(names)), "a conv runs twice: per-call ranges do not map to one Q/DQ"
    idx = list(range(len(stats)))
    res = {"frames": len(tok), "split": args.train_split, "names": names,
           "amax": {m: dict(zip(names, act_amax(stats, idx, m))) for m in METHODS}}
    (args.output / "calib.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({m: {"min": min(v.values()), "max": max(v.values())} for m, v in res["amax"].items()}), flush=True)


def nmse(a, b):
    return float(((a - b) ** 2).mean() / b.var())


def scorer(args):
    """score(amax) -> nMSE per output vs the float student on the val frames, and their mean."""
    w = fold_model(args)
    tok = val_tokens(args)
    ref = {t: [x.cpu() for x in w(images(args, t))[:5]] for t in tok}

    def score(a):
        sq.install(w.student, a)
        acc = {n: [] for n in OUT_NAMES}
        for t in tok:
            for n, x, b in zip(OUT_NAMES, w(images(args, t))[:5], ref[t]):
                acc[n].append(nmse(x.cpu(), b))
        sq.remove(w.student)
        r = {n: float(np.mean(v)) for n, v in acc.items()}
        r["mean"] = float(np.mean(list(r.values())))
        return r

    return score


@torch.no_grad()
def share(args):
    """amax["share_<level>"] per scale group, the better of the group's max / median in the sim."""
    cal = json.loads((args.output / "calib.json").read_text())
    base, score, res = cal["amax"][args.method], scorer(args), {"base": args.method}
    for level in ("block", "stage", "global"):
        r = {agg: score(sq.shared_amax(base, level, agg)) for agg in ("max", "median")}
        pick = min(r, key=lambda k: r[k]["mean"])
        cal["amax"][f"share_{level}"] = sq.shared_amax(base, level, pick)
        res[level] = {**r, "pick": pick, "groups": len(set(cal["amax"][f"share_{level}"].values()))}
        print(level, json.dumps(res[level]), flush=True)
    (args.output / "calib.json").write_text(json.dumps(cal, indent=1))
    (args.output / "share.json").write_text(json.dumps(res, indent=1))


@torch.no_grad()
def sim(args):
    cal = json.loads((args.output / "calib.json").read_text())
    amax = cal["amax"]
    score = scorer(args)
    res = {}
    for m in METHODS:
        res[m] = score(amax[m])
        print(m, json.dumps(res[m]), flush=True)
    # one global method cannot serve both branches (the vit tap wants the wide ranges, the levels the tight ones):
    # one pass of coordinate descent over the backbone's stages, starting from the best global method
    start = min(METHODS, key=lambda k: res[k]["mean"])
    best, best_r, chosen = dict(amax[start]), res[start], {}
    stage = lambda n: n.split(".")[1]                                # stem, stages_0 .. stages_3
    for g in sorted({stage(n) for n in best}):
        chosen[g] = start
        for m in METHODS:
            if m != start:
                cand = {n: amax[m][n] if stage(n) == g else best[n] for n in best}
                r = score(cand)
                print(g, m, json.dumps(r), flush=True)
                if r["mean"] < best_r["mean"]:
                    best, best_r, chosen[g] = cand, r, m
    res["hybrid"], res["hybrid_stages"], res["pick"] = best_r, chosen, "hybrid"
    amax["hybrid"] = best
    (args.output / "calib.json").write_text(json.dumps(cal, indent=1))
    (args.output / "sim.json").write_text(json.dumps(res, indent=1))
    print("pick hybrid", json.dumps(chosen), json.dumps(best_r), flush=True)


def qdq(args):
    import onnx

    a = json.loads((args.output / "calib.json").read_text())["amax"][args.method]
    m = onnx.load(str(args.onnx_in), load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    shapes = [list(inits[n.input[1]].dims) for n in convs]
    is_bb = [n.name.startswith("/backbone/") for n in convs]
    names = [sq.onnx_to_module(n.name) if b else None for n, b in zip(convs, is_bb)]
    del m
    assert sorted(n for n in names if n) == sorted(a), "ONNX backbone convs and calibrated convs differ"
    insert_qdq(args.onnx_in, args.onnx_out, [a[n] if n else 1.0 for n in names], shapes,
               skip=tuple(k for k, b in enumerate(is_bb) if not b))
    print(f"Q/DQ ({args.method}) on {sum(is_bb)} backbone convs -> {args.onnx_out}", flush=True)


def build(args):
    from qdistill.trt_utils import build_engine

    r = build_engine(args.onnx_out, "images", IMAGES, args.engine, precision="int8q", workspace_gb=2, opt_level=args.opt_level)
    print(json.dumps(r), flush=True)
    assert not r.get("failed"), r


@torch.no_grad()
def measure(args):
    from qdistill.trt_utils import time_engine

    head = load_head(args.weights)
    run, base = TrtStudentFPN(args.engine, head), TrtStudentFPN(args.fold_engine, head)
    tok = val_tokens(args)
    acc = {n: {"nmse": [], "cos": [], "nonfinite": 0} for n in OUT_NAMES}
    for t in tok:
        x = images(args, t)
        v, _ = run(x)
        got = [a.float().clone() for a in (v, *run.levels)]
        v, _ = base(x)
        for n, a, b in zip(OUT_NAMES, got, (v, *base.levels)):
            b = b.float()
            acc[n]["nmse"].append(nmse(a, b))
            acc[n]["cos"].append(float(F.cosine_similarity(a, b, dim=1 if n != "vit" else -1).mean()))
            acc[n]["nonfinite"] += int((~torch.isfinite(a)).sum())
    res = {"engine": str(args.engine), "frames": len(tok), "vs_fold_engine": {
        n: {"nmse": float(np.mean(v["nmse"])), "cos": float(np.mean(v["cos"])), "nonfinite": v["nonfinite"]} for n, v in acc.items()}}
    del run, base, head
    torch.cuda.empty_cache()
    res["latency_ms"] = {"engine": time_engine(args.engine), "fold": time_engine(args.fold_engine)}
    (args.output / f"measure_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def qat(args):
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    amax = json.loads((args.output / "calib.json").read_text())["amax"][args.method]
    w, ref = fold_model(args), fold_model(args)
    sq.install(w.student, amax)
    train_par = list(w.student.backbone.parameters())
    for q in train_par:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(train_par, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    train = split_tokens(args.splits, args.train_split)[:: args.train_stride]
    val = val_tokens(args)
    print(f"train {args.run_name}: {sum(q.numel() for q in train_par) / 1e6:.2f}M backbone params, {args.steps} steps of "
          f"{len(train)} frames, val {len(val)} frames, amax {args.method}", flush=True)

    def losses(t):
        x = images(args, t)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tgt = [a.float() for a in ref(x)[:5]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = w(x)[:5]
        Lo, total = {}, 0.0
        for n, a, b in zip(OUT_NAMES, out, tgt):
            a = a.float()
            nm = F.mse_loss(a, b) / b.var()
            cs = F.cosine_similarity(a, b, dim=1 if n != "vit" else -1).mean()
            Lo[f"{n}_nmse"], Lo[f"{n}_cos"] = nm, cs
            total = total + (nm + 1 - cs) / len(out)
        return total, Lo

    @torch.no_grad()
    def validate():
        acc = {}
        for t in val:
            loss, Lo = losses(t)
            for k, v in {"loss": loss, **Lo}.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        return {k: round(v / len(val), 5) for k, v in acc.items()}

    history = {"args": {k: str(v) for k, v in vars(args).items()}, "val_init": validate()}
    print(f"val init {json.dumps(history['val_init'])}", flush=True)
    order = np.random.default_rng(args.seed).permutation(len(train))
    t0, run_l = time.time(), {}
    for k in range(args.steps):
        loss, Lo = losses(train[order[k % len(train)]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_par, args.clip)
        opt.step()
        sched.step()
        for kk, vv in {"loss": loss.item(), **{a: b.item() for a, b in Lo.items() if a.endswith("nmse")}}.items():
            run_l[kk] = run_l.get(kk, 0.0) + vv
        if (k + 1) % args.log_every == 0 or k + 1 == args.steps:
            print(f"  step {k + 1}/{args.steps} " + " ".join(f"{a} {b / (k + 1):.4f}" for a, b in run_l.items())
                  + f" lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (k + 1):.2f} s/step"
                  + f" peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    history.update(val_final=validate(), min=round((time.time() - t0) / 60, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(f"val final {json.dumps(history['val_final'])}", flush=True)
    sq.remove(w.student)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck["model"] = {k: v.cpu() for k, v in w.student.state_dict().items()}
    ck["qat"] = {"method": args.method, "amax": amax}
    torch.save(ck, run / "final.pt")
    (run / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done {args.run_name} -> {run / 'final.pt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("calib", "sim", "share", "qdq", "build", "measure", "qat"), required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/e123")
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--method", default="hybrid", help="a key of calib.json amax (mse, pct, max, hybrid, share_<level>)")
    p.add_argument("--calib-frames", type=int, default=500)
    p.add_argument("--sample", type=int, default=1 << 18, help="calib: elements per tensor for the percentile / MSE clip")
    p.add_argument("--onnx-in", type=Path, default=ROOT / "outputs/e119/student_fold.onnx")
    p.add_argument("--onnx-out", type=Path, default=None)
    p.add_argument("--engine", type=Path, default=None)
    p.add_argument("--tag", default="ptq")
    p.add_argument("--opt-level", type=int, default=3)
    p.add_argument("--fold-engine", type=Path, default=ROOT / "weights/engines/student_front_int8.engine")
    p.add_argument("--run-name", default="qat")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-5, help="qat: ~3%% of the Stage-1 lr 3e-4")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-split", default="s1_C4k")
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--val-split", default="val_rest")
    p.add_argument("--val-stride", type=int, default=40)
    p.add_argument("--val-frames", type=int, default=30)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"calib": calib, "sim": sim, "share": share, "qdq": qdq, "build": build, "measure": measure, "qat": qat}[args.phase](args)


if __name__ == "__main__":
    main()
