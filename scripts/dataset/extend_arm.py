"""Extend a frame selection, keeping the earlier frames, so teacher outputs are reused."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from build_arms import RARE, interleave
from qdistill import metrics as M
from select_frames import DET_CLASS_NAMES, frame_stats

ROOT = Path(__file__).resolve().parents[2]
FAR_CLASSES = ("vehicle", "pedestrian")
FAR_RANGE = (20.0, 35.0)  # the 20-35 m bucket


def far_stats(gt_file: Path) -> Counter:
    """Boxes per class at FAR_RANGE (same box filter as frame_stats)."""
    gt = np.load(gt_file)
    keep = (np.abs(gt["boxes"][:, :2]) <= M.DET_RANGE).all(1) & (gt["num_pts"] > 0) & (gt["labels"] >= 0)
    r = np.hypot(gt["boxes"][keep, 0], gt["boxes"][keep, 1])
    far = (r > FAR_RANGE[0]) & (r <= FAR_RANGE[1])
    return Counter(DET_CLASS_NAMES[k] for k in gt["labels"][keep][far])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--embeddings", type=Path, default=ROOT / "data/splits/v1.0-trainval_dinov2_kept.npz")
    p.add_argument("--base", default="s1_C", help="frame list to extend, or 'none' to start empty")
    p.add_argument("--budget", type=int, required=True, help="total frames, base included")
    p.add_argument("--name", required=True)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--far-lam", type=float, default=0.0, help="weight of the far-box gain")
    p.add_argument("--compare", nargs="*", default=[], help="frame lists to report next to the result")
    p.add_argument("--report", type=Path, help="JSON file for the proxies")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device

    data = json.loads(args.splits.read_text())
    ranked, scenes, kept = data["train_ranked"], data["scenes"], data["frames"]
    base = [] if args.base == "none" else data["frame_lists"][args.base]
    group_of_scene = {n: scenes[n]["group"] for n in ranked}
    scene_of = {t: n for n in ranked for t in scenes[n]["samples"]}

    npz = np.load(args.embeddings)
    pool = [str(t) for t in npz["tokens"]]
    emb = torch.from_numpy(npz["emb"].astype(np.float32)).to(dev)
    row = {t: i for i, t in enumerate(pool)}
    assert all(t in row for t in base), "base frames must come from the kept pool"
    stats = {t: frame_stats(args.packed / t / "gt.npz")[0] for t in pool}
    far = {t: far_stats(args.packed / t / "gt.npz") for t in pool} if args.far_lam or args.report else {}

    chosen = set(base)
    acc, acc_far = Counter(), Counter()
    for t in base:
        acc.update(stats[t])
        if far:
            acc_far.update(far[t])
    min_dist = torch.full((len(pool),), 2.0, device=dev)
    for i in range(0, len(base), 1024):
        sel = emb[torch.tensor([row[t] for t in base[i:i + 1024]], device=dev)]
        min_dist = torch.minimum(min_dist, (1.0 - emb @ sel.T).min(1).values)

    def pick(g, cand):
        idx = torch.tensor([row[t] for t in cand], device=dev)
        dist = min_dist[idx].cpu().numpy()
        gain = np.array([sum(math.log1p(stats[t].get(c, 0) / (1 + acc[c])) for c in RARE) for t in cand])
        score = gain / max(gain.max(), 1e-9) + args.lam * dist / max(dist.max(), 1e-9)
        if args.far_lam:
            fg = np.array([sum(math.log1p(far[t].get(c, 0) / (1 + acc_far[c])) for c in FAR_CLASSES) for t in cand])
            score = score + args.far_lam * fg / max(fg.max(), 1e-9)
        t = cand[int(score.argmax())]
        acc.update(stats[t])
        if far:
            acc_far.update(far[t])
        min_dist.copy_(torch.minimum(min_dist, 1.0 - emb @ emb[row[t]]))
        return t

    remaining = defaultdict(list)
    for t in pool:
        if t not in chosen:
            remaining[group_of_scene[scene_of[t]]].append(t)
    start = Counter(group_of_scene[scene_of[t]] for t in base)
    new = interleave(dict(remaining), pick, limit=args.budget - len(base), start=start)
    out = list(base) + new

    pool_share = Counter(group_of_scene[scene_of[t]] for t in pool)
    cache = ROOT / "cache/teacher/v1.0-trainval"
    report = {}
    lists = [(args.base, base)] * bool(base) + [(c, data["frame_lists"][c]) for c in args.compare] + [(args.name, out)]
    for name, fr in lists:
        sel = emb[torch.tensor([row[t] for t in fr], device=dev)]
        nn = torch.cat([(1.0 - emb[i:i + 2048] @ sel.T).min(1).values for i in range(0, len(pool), 2048)])
        gmix = Counter(group_of_scene[scene_of[t]] for t in fr)
        rare = Counter()
        for t in fr:
            rare.update(stats[t])
        gdev = max(abs(gmix[g] / len(fr) - pool_share[g] / len(pool)) for g in pool_share)
        print(f"{name:8s} frames {len(fr)} scenes {len({scene_of[t] for t in fr})} grpdev {gdev:.3f} "
              f"cov mean {float(nn.mean()):.4f} p95 {float(nn.quantile(0.95)):.4f} rare "
              + " ".join(f"{c}={rare[c]}" for c in RARE))
        if far:
            fsum = Counter()
            for t in fr:
                fsum.update(far[t])
            uncached = sum(not (cache / "taps" / f"{t}.npz").exists()
                           and not (cache / "taps_vitW+llmInt8" / f"{t}.npz").exists() for t in fr)
            print(f"{'':8s} far {FAR_RANGE[0]:.0f}-{FAR_RANGE[1]:.0f} m " + " ".join(f"{c}={fsum[c]}" for c in DET_CLASS_NAMES)
                  + f" | all vehicle={rare['vehicle']} pedestrian={rare['pedestrian']} | uncached {uncached}")
            report[name] = {"frames": len(fr), "scenes": len({scene_of[t] for t in fr}), "max_group_share_dev": gdev,
                            "coverage_mean": float(nn.mean()), "coverage_p95": float(nn.quantile(0.95)),
                            "boxes_all": dict(rare), "boxes_far": dict(fsum), "uncached": uncached}
    cached = sum((cache / "taps" / f"{t}.npz").exists() for t in new)
    print(f"new frames {len(new)}, already cached (full taps, e.g. arm B) {cached}")
    data["frame_lists"][args.name] = out
    args.splits.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
