"""Thin nuScenes keyframes before selection."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from qdistill import metrics as M
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]
MOVING = ("vehicle.moving", "pedestrian.moving", "cycle.with_rider")


def frame_stats(gt_file: Path) -> tuple[Counter, int]:
    gt = np.load(gt_file)
    keep = (np.abs(gt["boxes"][:, :2]) <= M.DET_RANGE).all(1) & (gt["num_pts"] > 0) & (gt["labels"] >= 0)
    counts = Counter(DET_CLASS_NAMES[k] for k in gt["labels"][keep])
    moving = sum(any(m in a for m in MOVING) for a in gt["attributes"][keep])
    return counts, moving


def best_half(scores: list[float]) -> list[int]:
    """Exact DP: one index per pair (2k, 2k+1), no 'first of pair k' -> 'second of pair k+1' transition."""
    n = len(scores)
    pairs = [(i, i + 1) if i + 1 < n else (i,) for i in range(0, n, 2)]
    best = [{s: scores[idx] for s, idx in enumerate(pairs[0])}]
    back = [{}]
    for k in range(1, len(pairs)):
        cur, bk = {}, {}
        for s, idx in enumerate(pairs[k]):
            prev = {p: v for p, v in best[-1].items() if not (p == 0 and s == 1)}  # 1st -> next 2nd: 3-frame gap
            p = max(prev, key=prev.get)
            cur[s], bk[s] = prev[p] + scores[idx], p
        best.append(cur)
        back.append(bk)
    s = max(best[-1], key=best[-1].get)
    picks = []
    for k in range(len(pairs) - 1, -1, -1):
        picks.append(pairs[k][s])
        s = back[k].get(s, 0) if k else s
    return picks[::-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--moving-weight", type=float, default=0.5)
    args = p.parse_args()

    data = json.loads(args.splits.read_text())
    train = data["train_ranked"]
    stats = {t: frame_stats(args.packed / t / "gt.npz") for n in train for t in data["scenes"][n]["samples"]}
    total = Counter()
    for c, _ in stats.values():
        total.update(c)
    share = {c: total[c] / sum(total.values()) for c in total}
    w = {c: 1 / math.sqrt(share[c]) for c in share}
    w = {c: v / max(w.values()) for c, v in w.items()}

    def info(t):
        c, mv = stats[t]
        return sum(w[k] * math.log1p(v) for k, v in c.items()) + args.moving_weight * math.log1p(mv)

    frames, max_gap, kept_info, all_info = {}, 0, 0.0, 0.0
    kept_c, half_c = Counter(), Counter()
    for n in train:
        toks = data["scenes"][n]["samples"]
        sc = [info(t) for t in toks]
        picks = best_half(sc)
        assert len(picks) == math.ceil(len(toks) / 2)
        max_gap = max([max_gap] + [b - a for a, b in zip(picks, picks[1:])])
        frames[n] = [toks[i] for i in picks]
        kept_info += sum(sc[i] for i in picks)
        all_info += sum(sc)
        for i in picks:
            kept_c.update(stats[toks[i]][0])
        for i in range(0, len(toks), 2):  # baseline: every even frame
            half_c.update(stats[toks[i]][0])

    data["frames"] = frames
    data["frame_select"] = {"script": "scripts/dataset/select_frames.py", "class_weights": w,
                            "moving_weight": args.moving_weight, "max_gap_keyframes": 2}
    args.splits.write_text(json.dumps(data, indent=1))
    n_all, n_kept = len(stats), sum(map(len, frames.values()))
    print(f"train frames {n_all} -> kept {n_kept}; max gap {max_gap} keyframes; info kept {kept_info / all_info:.3f}")
    print("class weights:", {c: round(v, 3) for c, v in sorted(w.items(), key=lambda x: -x[1])})
    print(f"{'class':15s} {'all':>8s} {'kept':>8s} {'kept%':>6s} {'even%':>6s}")
    for c in sorted(total, key=lambda c: -w[c]):
        print(f"{c:15s} {total[c]:8d} {kept_c[c]:8d} {100 * kept_c[c] / total[c]:6.1f} {100 * half_c[c] / total[c]:6.1f}")


if __name__ == "__main__":
    main()
