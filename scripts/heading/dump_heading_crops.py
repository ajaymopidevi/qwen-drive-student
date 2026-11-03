"""Dump camera crops of pedestrian and bicycle boxes for the heading refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from qdistill import heading_crops as HC
from qdistill import metrics as M
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]
CLS = {4: "pedestrian", 2: "bicycle"}
MAX_DIST = 40.0
SHARD = 4000


def held_out(scene: str, frac: float) -> bool:
    return int(hashlib.md5(scene.encode()).hexdigest(), 16) % 1000 < frac * 1000


def frame_crops(job):
    """One frame -> list of (crop, meta)."""
    packed, token, boxes, labels, scores, idx, jitter, seed = job
    frame = PerceptionFrame(packed / token)
    rig = HC.Rig(frame)
    rng = np.random.default_rng(seed)
    views, imgs, out = [], {}, []
    for b, lab, sc, i in zip(boxes, labels, scores, idx):
        bj = b.copy()
        if jitter:
            bj[:2] += rng.uniform(-0.5, 0.5, 2)
            bj[3:6] *= rng.uniform(0.9, 1.1, 3)
        v = HC.best_view(rig, bj)
        if v is not None:
            views.append((b, bj, lab, sc, i, *v))
    for b, bj, lab, sc, i, c, rect in views:
        if c not in imgs:
            imgs[c] = np.asarray(frame.image(rig.cams[c]))
        out.append((HC.crop(imgs[c], rect), dict(alpha=HC.alpha(rig, b, c), cls=int(lab), dist=float(np.hypot(*b[:2])),
                                                 cam=c, token=token, box=int(i), score=float(sc))))
    return out


def write(out_dir: Path, prefix: str, items: list, n: int):
    x = np.stack([c for c, _ in items])
    np.save(out_dir / f"{prefix}_{n:03d}_x.npy", x)
    meta = {k: np.asarray([m[k] for _, m in items]) for k in items[0][1]}
    np.savez(out_dir / f"{prefix}_{n:03d}_meta.npz", **meta)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("gt", "pred"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pred", type=Path, default=None)
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--split", default="train_ranked")
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--holdout", type=float, default=0.10)
    p.add_argument("--min-score", type=float, default=0.1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = {"train": [], "hold": []} if args.mode == "gt" else {"pred": []}
    if args.mode == "gt":
        data = json.loads(args.splits.read_text())
        for scene in data[args.split]:
            part = "hold" if held_out(scene, args.holdout) else "train"
            for k, token in enumerate(data["scenes"][scene]["samples"][:: args.frame_stride]):
                g = np.load(args.packed / token / "gt.npz")
                b, lab = M.filter_gt({k2: g[k2] for k2 in g.files})
                keep = np.isin(lab, list(CLS)) & (np.hypot(b[:, 0], b[:, 1]) <= MAX_DIST)
                if keep.any():
                    seed = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
                    jobs[part].append((args.packed, token, b[keep], lab[keep], np.ones(keep.sum()), np.where(keep)[0],
                                       True, seed))
    else:
        d = json.loads(args.pred.read_text())
        assert d["class_names"][4] == "pedestrian" and d["class_names"][2] == "bicycle"
        for token, r in d["results"].items():
            if not r["boxes"]:
                continue
            b, lab, sc = np.asarray(r["boxes"]), np.asarray(r["labels"]), np.asarray(r["scores"])
            keep = np.isin(lab, list(CLS)) & (sc >= args.min_score) & (np.hypot(b[:, 0], b[:, 1]) <= MAX_DIST)
            if keep.any():
                jobs["pred"].append((args.packed, token, b[keep], lab[keep], sc[keep], np.where(keep)[0], False, 0))
    for part, js in jobs.items():
        js = js[: args.max_frames] if args.max_frames else js
        buf, n, total = [], 0, 0
        with Pool(args.workers) as pool:
            for i, items in enumerate(pool.imap(frame_crops, js, chunksize=4)):
                buf += items
                while len(buf) >= SHARD:
                    write(args.output, part, buf[:SHARD], n); total += SHARD; buf = buf[SHARD:]; n += 1
                if (i + 1) % 500 == 0:
                    print(f"{part}: {i + 1}/{len(js)} frames, {total + len(buf)} crops", flush=True)
        if buf:
            write(args.output, part, buf, n); total += len(buf); n += 1
        print(f"{part}: {len(js)} frames -> {total} crops in {n} shards ({args.output})", flush=True)


if __name__ == "__main__":
    main()
