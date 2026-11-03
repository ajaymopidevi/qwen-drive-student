"""Distill the 2D occupancy refiner from the teacher refiner."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdistill import narrow_refiner, occ2d
from qdistill.narrow_refiner import init_from_teacher
from qdistill.trt_utils import TrtRunner, build_engine, median_ms
from trt_occ_refiner import EMPTY, IN_SHAPE, OUT_SHAPE, agreement, load_refiner

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {**narrow_refiner.CONFIGS, **occ2d.CONFIGS}


def build(name_or_cfg) -> nn.Module:
    cfg = CONFIGS[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    return (occ2d if cfg.get("kind") == "occ2d" else narrow_refiner).build(cfg)


def export_onnx(model: nn.Module, path: Path):
    m = copy.deepcopy(model).float().cpu().eval()
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    if m.cfg.get("kind") == "occ2d":  # Z is folded into channels: fixed
        dummy, axes = torch.randn(1, 16, IN_SHAPE[2], 24, 24), {3: "Y", 4: "X"}
    else:
        dummy, axes = torch.randn(1, 16, 4, 24, 24), {2: "Z", 3: "Y", 4: "X"}
    torch.onnx.export(m, (dummy,), str(path), input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x": axes, "y": axes}, dynamo=False)


def n_params(m):
    return round(sum(p.numel() for p in m.parameters()) / 1e6, 2)


def val_files(args):
    """val (NNN_x.npy + stored NNN_y.npy) or a deployed-input cache (NNNNN.npy, targets from the engine)."""
    f = sorted(args.val_dir.glob("[0-9][0-9][0-9]_x.npy"))
    return f or sorted(args.val_dir.glob("[0-9][0-9][0-9][0-9][0-9].npy"))


def val_input(args, i=0):
    return torch.from_numpy(np.load(val_files(args)[i])).cuda().contiguous()


# ------------------------------------------------------------------------------------------------ arch
def phase_arch(args):
    d = args.output / "arch"
    d.mkdir(parents=True, exist_ok=True)
    x16 = val_input(args)
    res = {}
    for name in args.configs:
        m = build(name)
        eng = d / f"{name}_fp16.engine"
        if not eng.exists():
            export_onnx(m, d / f"{name}.onnx")
            r = build_engine(d / f"{name}.onnx", "x", IN_SHAPE, eng, "fp16", workspace_gb=args.workspace_gb)
            assert "engine" in r, r
        run = TrtRunner(eng, "x", IN_SHAPE, "y", OUT_SHAPE)
        ms = median_ms(lambda: run(x16), n=args.reps)
        res[name] = {"cfg": CONFIGS[name], "params_M": n_params(m), "median_ms": round(ms[0], 1), "p90_ms": round(ms[1], 1)}
        print(f"arch {name:12s} {res[name]['params_M']:6.2f}M  {ms[0]:6.1f} ms (p90 {ms[1]:.1f})", flush=True)
        del run
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------ cache
def phase_cache(args):
    from qdistill.student_io import load_student
    from qdistill.data import load_frame_images, split_tokens
    from qdistill.teacher import load_head, load_processor
    from qdistill.trt_install import TrtHead
    from qwen_drive_perception.dataset import PerceptionFrame

    toks = split_tokens(args.splits, args.train_split)[:: args.cache_stride][: args.cache_frames]
    d = args.cache_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames.json").write_text(json.dumps(toks))
    todo = [(i, t) for i, t in enumerate(toks) if not (d / f"{i:05d}.npy").exists()]
    print(f"cache: {len(toks)} frames, {len(todo)} to do -> {d}", flush=True)
    if not todo:
        return {"frames": len(toks), "dumped_now": 0, "dir": str(d)}
    student, norm = load_student(args.student, args.cache)
    head = load_head(args.weights)
    TrtHead(head, args.voxel_engine, None)
    bev = head.bev_modeling
    tr = bev.head.transformer
    tr.occ_decoder = None  # the exact refiner's weights are not needed here

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = None
            self.out = torch.zeros(OUT_SHAPE, dtype=torch.bfloat16, device="cuda")

        def forward(self, x):
            self.x = x.detach()
            return self.out

    stub = Stub()
    tr.occ_decoder = stub
    torch.cuda.empty_cache()
    processor = load_processor(args.weights)
    t0 = time.time()
    for k, (i, tok) in enumerate(todo):
        metas = processor(PerceptionFrame(args.packed / tok))[1]
        images = load_frame_images(args.packed, tok, "jpeg_gpu")[None].cuda()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vit_n, llm_n = student(images)
            vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])
            bev(img_vit_feats=vit.to(torch.bfloat16), img_llm_feats=llm.to(torch.bfloat16), img_metas=[metas])
        x = stub.x
        assert tuple(x.shape) == IN_SHAPE, x.shape
        tmp = d / f"{i:05d}.tmp.npy"
        np.save(tmp, x.to(torch.float16).cpu().numpy())
        tmp.rename(d / f"{i:05d}.npy")
        stub.x = None
        del x, vit, llm, vit_n, llm_n, images
        torch.cuda.empty_cache()  # the next frame otherwise OOMs in the voxel-pool kernel (as trt_occ_refiner calib)
        if k % 100 == 0 or k == len(todo) - 1:
            print(f"cache {k + 1}/{len(todo)} ({(time.time() - t0) / (k + 1):.2f} s/frame)", flush=True)
    return {"frames": len(toks), "dumped_now": len(todo), "dir": str(d), "s_per_frame": round((time.time() - t0) / len(todo), 2)}


# ------------------------------------------------------------------------------------------------ train
class Losses:
    def __init__(self, pred, kd_weight, occ_weight, feat_occ_weight=1.0):
        self.pred, self.kd_weight, self.occ_weight = pred, kd_weight, occ_weight
        self.feat_occ_weight = feat_occ_weight

    def logits(self, y):
        return self.pred(y.to(torch.bfloat16).permute(0, 4, 3, 2, 1)).float()  # [B, X, Y, Z, classes]

    def __call__(self, ys, yt, lt):
        yt32 = yt.float()
        if self.feat_occ_weight == 1.0:
            feat = F.mse_loss(ys.float(), yt32) / yt32.var().clamp_min(1e-6)
        else:  # voxels the teacher labels non-empty weighted feat_occ_weight x; lt is [B, X, Y, Z, K]
            occ = (lt.argmax(-1) != EMPTY).permute(0, 3, 2, 1)[:, None].float()          # [B, 1, Z, Y, X]
            wf = 1.0 + (self.feat_occ_weight - 1.0) * occ
            se = ((ys.float() - yt32) ** 2 * wf).sum() / (wf.sum() * ys.shape[1])
            feat = se / yt32.var().clamp_min(1e-6)
        ls = self.logits(ys)
        pt = lt.softmax(-1)
        kl = (pt * (lt.log_softmax(-1) - ls.log_softmax(-1))).sum(-1)
        w = 1.0 + (self.occ_weight - 1.0) * (lt.argmax(-1) != EMPTY).float()
        kl = (kl * w).sum() / w.sum()
        return feat + self.kd_weight * kl, {"feat": feat.item(), "kl": kl.item()}


def evaluate(model_fn, labels, args, n=None, target=None):
    """model_fn(x16) -> refiner output; agreement vs the reference labels: the stored PyTorch bf16 exact refiner
    outputs or, for a deployed-input cache, ``target(x16)`` (the --refiner-engine)."""
    files = val_files(args)[: n or None]
    acc = {}
    with torch.no_grad():
        for f in files:
            x16 = torch.from_numpy(np.load(f)).cuda().contiguous()
            if f.name.endswith("_x.npy"):
                y = torch.from_numpy(np.load(str(f).replace("_x.npy", "_y.npy"))).cuda()
            else:
                assert target is not None, f"{args.val_dir} has no stored outputs: needs the target engine"
                y = target(x16).clone()
            yp = model_fn(x16)
            yf = y.float()
            acc.setdefault("nmse", []).append(float(((yp.float() - yf) ** 2).mean() / yf.var().clamp_min(1e-12)))
            for k, v in agreement(labels(y), labels(yp)).items():
                acc.setdefault(k, []).append(v)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def phase_train(args):
    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    files = sorted(args.cache_dir.glob("[0-9][0-9][0-9][0-9][0-9].npy"))[: args.train_frames or None]
    assert files, f"no cached inputs in {args.cache_dir} (--phase cache)"
    torch.manual_seed(args.seed)
    teacher, pred = load_refiner(args)
    for p in pred.parameters():
        p.requires_grad_(False)
    model = build(args.config).cuda()
    init = init_from_teacher(model, teacher) if args.init == "teacher" else {}
    del teacher
    torch.cuda.empty_cache()
    target = TrtRunner(args.refiner_engine, "x", IN_SHAPE, "y", OUT_SHAPE)
    loss_fn = Losses(pred, args.kd_weight, args.occ_weight, args.feat_occ_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(files) if not args.max_steps else min(len(files), args.max_steps)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    start, history, best = 0, [], -1.0
    if args.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        print(f"resumed after epoch {ck['epoch']}", flush=True)
    print(f"train {args.config}: {n_params(model)}M params, init {init}, {len(files)} frames x {args.epochs} epochs",
          flush=True)

    def labels(y):
        return loss_fn.logits(y).argmax(-1)

    def student_fn(x16):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x16.float())

    if start == 0:
        model.eval()
        v0 = evaluate(student_fn, labels, args, n=args.val_frames, target=target)
        history.append({"epoch": -1, "val": v0})
        print(f"epoch -1 (init) val {json.dumps(v0)}", flush=True)
    rng = np.random.default_rng(args.seed)
    for epoch in range(start, args.epochs):
        model.train()
        order = rng.permutation(len(files))[:steps_per_epoch]
        t0, run_l = time.time(), {}
        # load the next frame while this one trains
        pool = ThreadPoolExecutor(1)
        nxt = pool.submit(np.load, files[order[0]])
        for k, j in enumerate(order):
            x = nxt.result()
            if k + 1 < len(order):
                nxt = pool.submit(np.load, files[order[k + 1]])
            x16 = torch.from_numpy(x).cuda().contiguous()
            with torch.no_grad():
                yt = target(x16).clone()
                lt = loss_fn.logits(yt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ys = model(x16.float())
            loss, parts = loss_fn(ys, yt, lt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            for kk, v in {"loss": loss.item(), **parts}.items():
                run_l[kk] = run_l.get(kk, 0.0) + v
            if (k + 1) % args.log_every == 0:
                print(f"  ep {epoch} step {k + 1}/{steps_per_epoch} " + " ".join(f"{kk} {v / (k + 1):.4f}" for kk, v in run_l.items())
                      + f" lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (k + 1):.2f} s/step", flush=True)
        pool.shutdown()
        model.eval()
        v = evaluate(student_fn, labels, args, n=args.val_frames, target=target)
        tr = {kk: vv / steps_per_epoch for kk, vv in run_l.items()}
        history.append({"epoch": epoch, "train": tr, "val": v, "min": round((time.time() - t0) / 60, 1)})
        print(f"epoch {epoch} train {json.dumps(tr)} val {json.dumps(v)}", flush=True)
        ck = {"model": model.state_dict(), "cfg": model.cfg, "config": args.config, "epoch": epoch, "val": v,
              "args": {k: str(v) for k, v in vars(args).items()}}
        if v["agree_occupied"] > best:
            best = v["agree_occupied"]
            torch.save(ck, run / "best.pt")
        torch.save({**ck, "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history, "best": best},
                   run / "last.pt")
    out = {"config": args.config, "params_M": n_params(model), "init": init, "history": history, "best_agree_occupied": best,
           "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(f"peak {out['peak_GiB']} GiB", flush=True)
    (run / "history.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ build
def phase_build(args):
    run = ROOT / "runs" / args.run_name
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build(ck["cfg"])
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    onnx_path = args.output / f"{args.run_name}.onnx"
    eng = args.output / f"{args.run_name}_fp16.engine"
    if not eng.exists():
        export_onnx(model, onnx_path)
        r = build_engine(onnx_path, "x", IN_SHAPE, eng, "fp16", workspace_gb=args.workspace_gb)
        assert "engine" in r, r
    _, pred = load_refiner(args)
    labels = lambda y: pred(y.to(torch.bfloat16).permute(0, 4, 3, 2, 1)).argmax(-1)
    exact = TrtRunner(args.refiner_engine, "x", IN_SHAPE, "y", OUT_SHAPE)
    acc_pt = evaluate(lambda x: _bf16(model, x), labels, args, target=exact)
    del model
    torch.cuda.empty_cache()
    runner = TrtRunner(eng, "x", IN_SHAPE, "y", OUT_SHAPE)
    acc_trt = evaluate(lambda x: runner(x), labels, args, target=exact)
    x16 = val_input(args)
    ms = median_ms(lambda: runner(x16))
    ms_exact = median_ms(lambda: exact(x16))
    acc_exact = evaluate(lambda x: exact(x), labels, args, target=exact)
    out = {"run": args.run_name, "epoch": ck["epoch"], "engine": str(eng), "target_engine": str(args.refiner_engine),
           "val_dir": str(args.val_dir), "pytorch_bf16": acc_pt, "trt_fp16": acc_trt,
           "exact_trt_fp16": acc_exact, "median_ms": round(ms[0], 1), "p90_ms": round(ms[1], 1),
           "exact_median_ms": round(ms_exact[0], 1)}
    print(json.dumps(out, indent=1), flush=True)
    return out


def _bf16(model, x16):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x16.float())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["arch", "cache", "train", "build"], required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/engines")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   help="arch: candidates (qdistill.narrow_refiner / qdistill.occ2d CONFIGS)")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--config", default="c32_64_b1")
    p.add_argument("--run-name", default=None)
    p.add_argument("--init", choices=["teacher", "random"], default="teacher")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--occ-weight", type=float, default=4.0, help="KL weight of voxels the teacher labels non-empty")
    p.add_argument("--feat-occ-weight", type=float, default=1.0, help="feature-MSE weight of those voxels")
    p.add_argument("--train-frames", type=int, default=0, help="use the first n cached frames (0 = all)")
    p.add_argument("--val-frames", type=int, default=0, help="val on the first n of the 101 frames (0 = all)")
    p.add_argument("--max-steps", type=int, default=0, help="smoke: steps per epoch")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-split", default="s1_C4k")
    p.add_argument("--cache-frames", type=int, default=2000)
    p.add_argument("--cache-stride", type=int, default=2)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache/refiner_in")
    p.add_argument("--val-dir", type=Path, default=ROOT / "cache/refiner_val",
                   help="val (stored outputs) or a deployed-input cache such as cache/val_rest_in")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--workspace-gb", type=int, default=2)
    p.add_argument("--student", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("train", "build"):
        assert args.run_name, "--run-name"
    out = {"arch": phase_arch, "cache": phase_cache, "train": phase_train, "build": phase_build}[args.phase](args)
    name = args.phase if args.phase in ("arch", "cache") else f"{args.run_name}_{args.phase}"
    (args.output / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"-> {args.output / f'{name}.json'}", flush=True)


if __name__ == "__main__":
    main()
