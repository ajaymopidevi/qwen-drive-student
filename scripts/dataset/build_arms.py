"""Build frame selection arms from DINOv2 embeddings (rare-class gain + k-center)."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from select_frames import frame_stats

ROOT = Path(__file__).resolve().parents[2]
RARE = ("bicycle", "generic_object", "traffic_cone", "barrier")


def interleave(groups: dict[str, list[str]], pick, limit: int | None = None, start: Counter | None = None) -> list[str]:
    """Group-proportional interleaving; ``pick(group, candidates)`` chooses within the group.

    ``start``: per-group counts already selected; quotas count them, ``limit``
    is the number of new picks. ``groups`` must then hold only the not-yet-selected candidates.
    """
    total = sum(map(len, groups.values())) + sum((start or Counter()).values())
    share = {g: (len(v) + (start or Counter())[g]) / total for g, v in groups.items()}
    remaining = {g: list(v) for g, v in groups.items()}
    taken, order = Counter(start or {}), []
    n0 = sum(taken.values())
    while any(remaining.values()) and (limit is None or len(order) < limit):
        k = n0 + len(order) + 1
        g = max((g for g in groups if remaining[g]),
                key=lambda g: (max(share[g] * k, 1.0 if taken[g] == 0 else 0.0) - taken[g], share[g]))
        x = pick(g, remaining[g])
        remaining[g].remove(x)
        taken[g] += 1
        order.append(x)
    return order


def take_scenes(order, frames_of, budget) -> list[str]:
    out = []
    for n in order:
        if len(out) >= budget:
            break
        out += frames_of(n)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--embeddings", type=Path, default=ROOT / "data/splits/v1.0-trainval_dinov2_kept.npz")
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=Path, default=ROOT / "outputs/stage1_arms.json")
    args = p.parse_args()

    data = json.loads(args.splits.read_text())
    ranked, scenes, kept = data["train_ranked"], data["scenes"], data["frames"]
    group_of_scene = {n: scenes[n]["group"] for n in ranked}
    scene_of = {t: n for n in ranked for t in scenes[n]["samples"]}
    stats = {t: frame_stats(args.packed / t / "gt.npz") for n in ranked for t in scenes[n]["samples"]}

    npz = np.load(args.embeddings)
    pool = [str(t) for t in npz["tokens"]]
    assert set(pool) == {t for n in ranked for t in kept[n]}, "embeddings must cover the kept train pool"
    emb = torch.from_numpy(npz["emb"].astype(np.float32)).cuda()
    row = {t: i for i, t in enumerate(pool)}

    # A: stratified random scenes; B: ranking; D: ranking, all frames.
    rng = random.Random(args.seed)
    by_group = defaultdict(list)
    for n in ranked:
        by_group[group_of_scene[n]].append(n)
    order_a = interleave({g: sorted(v) for g, v in by_group.items()}, lambda g, c: rng.choice(c))
    arms = {
        "A": take_scenes(order_a, lambda n: kept[n], args.budget),
        "B": take_scenes(ranked, lambda n: kept[n], args.budget),
        "D": take_scenes(ranked, lambda n: scenes[n]["samples"], args.budget),
    }

    # C: frame-level greedy over the pool.
    frames_by_group = defaultdict(list)
    for t in pool:
        frames_by_group[group_of_scene[scene_of[t]]].append(t)
    acc = Counter()
    min_dist = torch.full((len(pool),), 2.0, device="cuda")

    def pick_c(g, cand):
        idx = torch.tensor([row[t] for t in cand], device="cuda")
        dist = min_dist[idx].cpu().numpy()
        gain = np.array([sum(math.log1p(stats[t][0].get(c, 0) / (1 + acc[c])) for c in RARE) for t in cand])
        score = gain / max(gain.max(), 1e-9) + args.lam * dist / max(dist.max(), 1e-9)
        t = cand[int(score.argmax())]
        acc.update(stats[t][0])
        min_dist.copy_(torch.minimum(min_dist, 1.0 - emb @ emb[row[t]]))
        return t

    arms["C"] = interleave(frames_by_group, pick_c, limit=args.budget)
    arms = {k: arms[k] for k in "ABCD"}

    # Proxies.
    pool_share = Counter(group_of_scene[scene_of[t]] for t in pool)
    total_rare = Counter()
    for t in pool:
        total_rare.update(stats[t][0])
    report = {}
    for name, fr in arms.items():
        # D holds whole scenes; its non-kept frames have no embedding, so its kept frames stand in for them.
        rows = sorted({row[k] for n in {scene_of[t] for t in fr} for k in kept[n]} if name == "D" else {row[t] for t in fr})
        sel = emb[torch.tensor(rows, device="cuda")]
        nn_dist = torch.cat([(1.0 - emb[i:i + 2048] @ sel.T).min(1).values for i in range(0, len(pool), 2048)])
        gmix = Counter(group_of_scene[scene_of[t]] for t in fr)
        rare = Counter()
        for t in fr:
            rare.update(stats[t][0])
        report[name] = {
            "frames": len(fr), "scenes": len({scene_of[t] for t in fr}),
            "max_group_share_dev": max(abs(gmix[g] / len(fr) - pool_share[g] / len(pool)) for g in pool_share),
            "rare_boxes": {c: rare[c] for c in RARE},
            "coverage_mean": float(nn_dist.mean()), "coverage_p95": float(nn_dist.quantile(0.95)),
        }
    union = list(dict.fromkeys(t for fr in arms.values() for t in fr))
    data.setdefault("frame_lists", {}).update({f"s1_{k}": v for k, v in arms.items()} | {"s1_union": union})
    args.splits.write_text(json.dumps(data, indent=1))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"budget": args.budget, "lam": args.lam, "seed": args.seed,
                                       "union_frames": len(union), "arms": report}, indent=2))

    print(f"union {len(union)} frames")
    print(f"{'arm':4s} {'frames':>6s} {'scenes':>6s} {'grpdev':>6s} {'covMean':>8s} {'covP95':>7s} " + " ".join(f"{c[:7]:>8s}" for c in RARE))
    for k, r in report.items():
        print(f"{k:4s} {r['frames']:6d} {r['scenes']:6d} {r['max_group_share_dev']:6.3f} {r['coverage_mean']:8.4f} "
              f"{r['coverage_p95']:7.4f} " + " ".join(f"{r['rare_boxes'][c]:8d}" for c in RARE))


if __name__ == "__main__":
    main()
