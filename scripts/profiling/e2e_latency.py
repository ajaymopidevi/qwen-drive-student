"""End-to-end latency of a chain on val_quick: median per stage and total."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.autograd.profiler import record_function

from qdistill.student_io import MODULES, get, load_student
from qdistill.data import load_frame_images, load_images_gpu, split_tokens
from qdistill.light_blocks import install_light_voxel, load_light_voxel
from qdistill.teacher import load_head, load_processor
from qdistill.trt_install import TrtHead
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("pt_e038", "trt_vox", "trt_all")


class Timer:
    """Synchronised wall-clock ms per named module (forward pre / post hooks), summed per frame."""

    def __init__(self, targets: list[tuple[str, torch.nn.Module]]):
        self.frame, self.start, self.times = defaultdict(float), {}, defaultdict(list)
        self.handles = []
        for name, m in targets:
            self.handles.append(m.register_forward_pre_hook(lambda mod, a: self._pre(mod)))
            self.handles.append(m.register_forward_hook(lambda mod, a, o, name=name: self._post(mod, name)))

    def _pre(self, mod):
        torch.cuda.synchronize()
        self.start[id(mod)] = time.perf_counter()

    def _post(self, mod, name):
        torch.cuda.synchronize()
        self.frame[name] += 1000 * (time.perf_counter() - self.start.pop(id(mod)))

    def end_frame(self, keep: bool):
        if keep:
            for k, v in self.frame.items():
                self.times[k].append(v)
        self.frame = defaultdict(float)

    def remove(self):
        for h in self.handles:
            h.remove()


POST = None  # --fast-post: qdistill.fast_postproc.FastPost (one host sync for the read-out)


def deploy_outputs(bev, vit, llm, metas) -> dict:
    """What a deployment reads out: boxes, occupancy labels, map labels on the CPU (run_head also copies raw logits
    for distillation, ~15 MB per frame, which a deployment does not need)."""
    with record_function("head forward"):
        outs = bev(img_vit_feats=vit.to(torch.bfloat16), img_llm_feats=llm.to(torch.bfloat16), img_metas=[metas])
    with record_function("head postproc (boxes, labels to CPU)"):
        if POST is not None:
            return POST(bev.head, outs, metas)
        det = bev.head.get_bboxes(outs, [metas])[0]
        return {"det": det, "occ": bev.head.get_occ(outs, [metas])[0].byte().cpu().numpy(),
                "map": bev.head.get_map_seg(outs, [metas])[0].byte().cpu().numpy()}


def _det_arrays(det) -> list:
    """get_bboxes output for one frame -> its tensors / arrays (boxes, scores, labels), whatever the container."""
    items = det if isinstance(det, (list, tuple)) else list(det.values()) if isinstance(det, dict) else [det]
    out = []
    for x in items:
        if hasattr(x, "tensor"):
            x = x.tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            out.append(x.detach().float().cpu())
    return out


def compare(ref: dict, got: dict) -> dict:
    """Fast path vs reference outputs on the same frames: label equality and det max abs diff (boxes + scores)."""
    occ = [float((ref[t]["occ"] == got[t]["occ"]).mean()) for t in ref]
    mp = [float((ref[t]["map"] == got[t]["map"]).mean()) for t in ref]
    det = []
    for t in ref:
        a, b = _det_arrays(ref[t]["det"]), _det_arrays(got[t]["det"])
        det.append(max((float((x - y).abs().max()) if x.shape == y.shape and x.numel() else float("inf") if x.shape != y.shape else 0.0)
                       for x, y in zip(a, b)) if a else float("nan"))
    return {"frames": len(ref), "occ_equal_min": min(occ), "occ_equal_mean": float(np.mean(occ)),
            "map_equal_min": min(mp), "map_equal_mean": float(np.mean(mp)),
            "det_max_abs_diff_max": max(det), "det_max_abs_diff_median": float(np.median(det))}


def stats(xs: list[float]) -> dict:
    return {"median_ms": round(float(np.median(xs)), 1), "p90_ms": round(float(np.percentile(xs, 90)), 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/latency")
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--light-voxel", type=Path, default=ROOT / "weights/voxel_r64.pt")
    p.add_argument("--voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--merged-pool", action="store_true",
                   help="merged-camera voxel pooling straight into the voxel engine (bit-identical input)")
    p.add_argument("--chain", default=None, help="head_chain spec timed as deployed (e.g. skip_ca:3,skip_ca:0)")
    p.add_argument("--heading", type=Path, default=None, help="heading refiner checkpoint (runs/<run>/best.pt)")
    p.add_argument("--heading-engine", type=Path, default=ROOT / "weights/engines/heading_refiner_fp16_b64.engine")
    p.add_argument("--student-engine", type=Path, default=None,
                   help="fused student + de-normalisation + LLM adaptor engine (trt_student_fpn.py); timed as student")
    p.add_argument("--frames", type=int, default=25, help="val_quick every 48th frame, this many")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--graph-decoder", action="store_true",
                   help="the sampling decoder as a CUDA graph (qdistill.graph_decoder); outputs vs the "
                        "eager decoder on the same frames -> graph_vs_reference")
    p.add_argument("--graph-head", action="store_true",
                   help="A: the head after voxel pooling as one CUDA graph (qdistill.graph_head; needs "
                        "--graph-decoder); outputs vs the graphed-decoder head on the same frames -> graph_head_vs_reference")
    p.add_argument("--fast-post", action="store_true",
                   help="A: read-out with one host sync (qdistill.fast_postproc); outputs vs upstream get_bboxes / "
                        "get_occ / get_map_seg on the same frames -> fast_post_vs_reference")
    p.add_argument("--sync-trim", action="store_true",
                   help="A: voxel pooling with fewer host syncs (qdistill.sync_trim; needs --merged-pool and "
                        "--chain ...fast_geom); outputs vs reference -> sync_trim_vs_reference")
    p.add_argument("--side-stream", action="store_true",
                   help="A: run each frame on a non-default CUDA stream (TensorRT adds syncs on the default "
                        "stream); outputs vs reference -> side_stream_vs_reference")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    student, norm = load_student(args.ckpt, args.cache)
    head = load_head(args.weights)
    bev = head.bev_modeling
    extra = []  # (name, module) timed inside view_trans
    info = {}
    if args.variant == "pt_e038":
        block = load_light_voxel(args.light_voxel)
        light = install_light_voxel(head, block)  # keep the reference: a collected context manager uninstalls
        light.__enter__()
        extra.append(("  voxel block", block))
    else:
        trt = TrtHead(head, args.voxel_engine, args.refiner_engine if args.variant == "trt_all" else None)
        info = trt.info()
        if args.merged_pool:
            from qdistill import merged_pool
            merged_pool.install(head, trt)
            info["merged_pool"] = True
        extra.append(("  voxel block", trt.voxel))
    if args.chain:
        from qdistill import head_chain
        info["ablate"] = head_chain.install(head, args.chain)
        vt = bev.view_trans  # the block engine that actually runs
        eng = next((getattr(vt, a) for a in ("_zhalf_engine", "_warp_engine") if hasattr(vt, a)), None)
        if eng is not None:
            extra = [(n, eng if n == "  voxel block" else m) for n, m in extra]
    sfpn = None
    if args.student_engine is not None:
        from qdistill import trt_student
        sfpn = trt_student.load(args.student_engine, head)
        info["student_fpn"] = {"engine": str(args.student_engine), **sfpn.info()}
    processor = load_processor(args.weights)

    tokens = split_tokens(args.splits, "val_quick")[::48][: args.frames]
    rig = None
    metas, hrigs, hr = {}, {}, None
    if args.heading is not None:
        from qdistill.heading_gpu import HeadingGPU
        hr = HeadingGPU(args.heading, args.heading_engine)
        info["heading"] = {"ckpt": str(args.heading), "engine": str(args.heading_engine), "tau": hr.tau}
    for t in tokens:
        f = PerceptionFrame(args.packed / t)
        metas[t] = processor(f)[1] if rig is None else rig.apply(processor(f)[1], t, f.cam_intrinsic)
        if hr is not None:
            hrigs[t] = hr.rig(f)

    side = {"stream": None}

    def frame(token, timer=None, keep=None):
        """One frame end to end (on the side stream when --side-stream is on); returns ms per part."""
        if side["stream"] is None:
            return _frame(token, timer, keep)
        with torch.cuda.stream(side["stream"]):
            return _frame(token, timer, keep)

    def _frame(token, timer=None, keep=None):
        """One frame end to end; returns ms for decode / student / head / total."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if hr is None:
            images = load_frame_images(args.packed, token, "jpeg_gpu")[None].cuda()
        else:
            images, raw = load_images_gpu(args.packed / token, return_raw=True)
            images = images[None]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if sfpn is not None:  # student + adaptor in one engine; the head's adaptor returns its levels
            with torch.no_grad():
                vit, llm = sfpn(images)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vit_n, llm_n = student(images)
            vit, llm = norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.no_grad(), record_function("head"):
            o = deploy_outputs(bev, vit, llm, metas[token])
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        r = {"decode": 1000 * (t1 - t0), "student": 1000 * (t2 - t1), "head": 1000 * (t3 - t2)}
        if hr is not None:
            hr(o["det"], raw, hrigs[token])
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            r["heading"], r["heading_crops"] = 1000 * (t4 - t3), hr.last["cropped"]
            t3 = t4
        if keep is not None:
            keep[token] = o
        if timer is not None:
            timer.end_frame(True)
        return {**r, "total": 1000 * (t3 - t0)}

    agreement, fast = None, None
    gdec, gagree = None, None
    if args.graph_decoder:
        from qdistill import graph_decoder
        ref = {}
        for tok in tokens:
            frame(tok, keep=ref)
        gdec = graph_decoder.install(head)
        got = {}
        for tok in tokens:
            frame(tok, keep=got)
        gagree = compare(ref, got)
        info["graph_decoder"] = gdec.info()
        print("graph decoder vs reference:", json.dumps(gagree), json.dumps(gdec.info()), flush=True)
    ghagree = None
    if args.graph_head:
        from qdistill import graph_head
        assert gdec is not None, "--graph-head needs --graph-decoder"
        ref = {}
        for tok in tokens:
            frame(tok, keep=ref)
        gh = graph_head.install(head)
        got = {}
        for tok in tokens:
            frame(tok, keep=got)
        ghagree = compare(ref, got)
        info["graph_head"] = gh.info()
        print("graph head vs reference:", json.dumps(ghagree), json.dumps(gh.info()), flush=True)

    fgagree = None

    stagree = None
    if args.sync_trim:
        from qdistill import sync_trim
        ref = {}
        for tok in tokens:
            frame(tok, keep=ref)
        info.update(sync_trim.install(head))
        got = {}
        for tok in tokens:
            frame(tok, keep=got)
        stagree = compare(ref, got)
        print("sync trim vs reference:", json.dumps(stagree), flush=True)
    ssagree = None
    if args.side_stream:
        ref = {}
        for tok in tokens:
            frame(tok, keep=ref)
        side["stream"] = torch.cuda.Stream()
        got = {}
        for tok in tokens:
            frame(tok, keep=got)
        ssagree = compare(ref, got)
        print("side stream vs reference:", json.dumps(ssagree), flush=True)
    fpagree = None
    if args.fast_post:
        global POST
        from qdistill.fast_postproc import FastPost
        ref = {}
        for tok in tokens:
            frame(tok, keep=ref)
        POST = FastPost()
        got = {}
        for tok in tokens:
            frame(tok, keep=got)
        fpagree = compare(ref, got)
        print("fast post vs reference:", json.dumps(fpagree), flush=True)

    torch.cuda.reset_peak_memory_stats()
    parts = defaultdict(list)
    for i, tok in enumerate(tokens):
        r = frame(tok)
        if i >= args.warmup:
            for k, v in r.items():
                parts[k].append(v)
    peak = torch.cuda.max_memory_allocated() / 2**30

    targets = [(name, get(bev, pth)) for name, path in MODULES.items() for pth in (path if isinstance(path, tuple) else (path,))]
    timer = Timer(targets + extra)
    hooked = []
    for i, tok in enumerate(tokens):
        r = frame(tok)
        timer.end_frame(i >= args.warmup)
        if i >= args.warmup:
            hooked.append(r["head"])
    timer.remove()
    modules = {k: stats(v) for k, v in timer.times.items()}
    if "voxel_pool (view_trans)" in modules:  # view_trans = pooling (coords + kernel + camera sum) + voxel block
        inner = "  voxel block"
        assert timer.times[inner], f"{inner!r} never timed: its module is not the one the chain calls"
        pool = np.array(timer.times["voxel_pool (view_trans)"]) - np.array(timer.times[inner])
        modules["  voxel pooling (coords, kernel, cam sum)"] = stats(list(pool))
    top = sum(np.median(v) for k, v in timer.times.items() if not k.startswith("  "))
    out = {
        "variant": args.variant, "frames": len(tokens) - args.warmup, "warmup": args.warmup,
        "end_to_end": {k: stats(v) for k, v in parts.items() if k != "heading_crops"},
        "heading_crops_per_frame": {"median": float(np.median(parts["heading_crops"])), "max": int(max(parts["heading_crops"]))}
        if "heading_crops" in parts else None,
        "heading_profile_ms": {k: round(float(np.median(v)), 3) for k, v in hr.prof.items()}
        if hr is not None and getattr(hr, "prof", None) else None,
        "head_hooked_median_ms": round(float(np.median(hooked)), 1),
        "modules": modules,
        "unattributed_ms": round(float(np.median(hooked) - top), 1),
        "peak_GiB": round(peak, 2), "trt": info,
        "fast_vs_reference": agreement,
        "graph_vs_reference": gagree, "graph_head_vs_reference": ghagree, "fast_geom_vs_reference": fgagree,
        "fast_post_vs_reference": fpagree, "sync_trim_vs_reference": stagree, "side_stream_vs_reference": ssagree,
        "note": "img_metas prepared before the clock; head = forward + boxes / occ / map labels to the CPU",
    }
    path = args.output / f"latency_{args.variant}.json"
    if hr is not None:
        path = path.with_name(path.stem + "_heading.json")
    if args.graph_decoder:
        path = path.with_name(path.stem + "_gdec.json")
    if args.graph_head:
        path = path.with_name(path.stem + "_ghead.json")
    if args.fast_post:
        path = path.with_name(path.stem + "_fpost.json")
    if args.sync_trim:
        path = path.with_name(path.stem + "_strim.json")
    if args.side_stream:
        path = path.with_name(path.stem + "_side.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
