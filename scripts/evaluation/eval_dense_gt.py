"""Map and occupancy mIoU of saved dumps against GT, with a paired scene bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qdistill.np_io import load_npz
from qdistill.data import split_tokens

ROOT = Path(__file__).resolve().parents[2]
MAP_CLASSES = ("background", "driveable_surface", "road_line", "road_edge", "crosswalk", "walkway")
OCC_CLASSES = ("vehicle", "czone_sign", "bicycle", "generic_object", "pedestrian", "traffic_cone", "barrier",
               "driveable", "background", "empty")
OCC_EVAL = (0, 2, 3, 4, 5, 6, 7, 8)  # classes with Occ3D GT
OCC_EMPTY = 9
# Occ3D: 0 others, 1 barrier, 2 bicycle, 3 bus, 4 car, 5 construction_vehicle, 6 motorcycle, 7 pedestrian,
# 8 traffic_cone, 9 trailer, 10 truck, 11 driveable_surface, 12 other_flat, 13 sidewalk, 14 terrain, 15 manmade,
# 16 vegetation, 17 free. Vehicle / bicycle grouping as nuscenes_pack.LABEL_MAP.
OCC3D_TO_TEACHER = np.array([3, 6, 2, 0, 0, 0, 2, 4, 5, 0, 0, 7, 8, 8, 8, 8, 8, 9], np.uint8)
X_BINS = {"fwd 0-10 m": (0, 10), "fwd 10-20 m": (10, 20), "fwd 20-30 m": (20, 30),
          "back 0-10 m": (-10, 0), "back 10-20 m": (-20, -10), "back 20-30 m": (-30, -20)}
XS = -30 + 0.15 * (np.arange(400) + 0.5)


def counts(pred: np.ndarray, gt: np.ndarray, n: int, mask=None) -> np.ndarray:
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    c = np.bincount(gt.ravel().astype(np.int64) * n + pred.ravel(), minlength=n * n).reshape(n, n)
    inter = np.diag(c)
    return np.stack([inter, c.sum(0) + c.sum(1) - inter], 1).astype(np.float64)  # [n, (inter, union)]


def ious(c: np.ndarray, names, keep) -> dict:
    per = {names[k]: (c[k, 0] / c[k, 1] if c[k, 1] else None) for k in keep}
    vals = [v for v in per.values() if v is not None]
    return {"per_class": per, "miou": float(np.mean(vals)) if vals else None}


def who_right(s: np.ndarray, t: np.ndarray, g: np.ndarray, mask=None) -> np.ndarray:
    d = s != t
    if mask is not None:
        d &= mask
    return np.array([d.sum(), (d & (s == g)).sum(), (d & (t == g)).sum()], np.float64)


def bootstrap(a: dict, b: dict, keep, reps: int, rng) -> dict:
    scenes = sorted(set(a) & set(b))
    A, B = np.array([a[s] for s in scenes]), np.array([b[s] for s in scenes])

    def stat(idx):
        ia, ib = A[idx].sum(0), B[idx].sum(0)
        ca = ia[keep, 0] / np.maximum(ia[keep, 1], 1)
        cb = ib[keep, 0] / np.maximum(ib[keep, 1], 1)
        return np.concatenate([[cb.mean() - ca.mean()], cb - ca])

    full = stat(np.arange(len(scenes)))
    draws = np.array([stat(rng.integers(0, len(scenes), len(scenes))) for _ in range(reps)])
    return {"miou_diff": float(full[0]), "ci95": [float(np.percentile(draws[:, 0], 2.5)), float(np.percentile(draws[:, 0], 97.5))],
            "per_class_diff": [float(x) for x in full[1:]]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", type=Path, default=ROOT / "outputs/dumps")
    p.add_argument("--models", nargs="*")
    p.add_argument("--split", default="val_quick")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--map-gt", type=Path, default=ROOT / "data/gt/map")
    p.add_argument("--occ-gt", type=Path, default=ROOT / "data/occ3D_nuScenes/gts")
    p.add_argument("--poses", type=Path, default=ROOT / "data/gt/ego_poses_val.json")
    p.add_argument("--baseline", default=None)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--output", type=Path, default=ROOT / "data/gt/dense_gt_eval.json")
    args = p.parse_args()

    scene_of = {t: v["scene"] for t, v in json.loads(args.poses.read_text()).items()}
    tokens = split_tokens(args.splits, args.split)[:: args.stride]
    models = args.models if args.models is not None else sorted(d.name for d in args.dumps.iterdir() if d.is_dir())
    models = [m for m in models if all((args.dumps / m / f"{t}.npz").exists() for t in tokens)]
    has_teacher = all((args.cache / "head" / f"{t}.npz").exists() for t in tokens)
    names = (["teacher"] if has_teacher else []) + models
    print(f"{len(tokens)} frames; models: {names}", flush=True)
    acc = {n: {"map": np.zeros((6, 2)), "occ": np.zeros((10, 2)), "geo": np.zeros((2, 2)),
               "map_x": {k: np.zeros((6, 2)) for k in X_BINS}, "map_scene": {}, "occ_scene": {},
               "map_who": np.zeros(3), "occ_who": np.zeros(3)} for n in names}
    xmask = {k: np.broadcast_to((XS >= a) & (XS < b), (200, 400)) for k, (a, b) in X_BINS.items()}
    for i, tok in enumerate(tokens):
        sc = scene_of[tok]
        gmap = np.load(args.map_gt / f"{tok}.npy")
        o3 = np.load(args.occ_gt / sc / tok / "labels.npz")
        gocc = OCC3D_TO_TEACHER[o3["semantics"]]
        cam = o3["mask_camera"].astype(bool)
        preds = {}
        if has_teacher:
            teacher = load_npz(args.cache / "head" / f"{tok}.npz")
            preds["teacher"] = (teacher["map"].astype(np.uint8), teacher["occ"].astype(np.uint8))
        for m in models:
            z = np.load(args.dumps / m / f"{tok}.npz")
            preds[m] = (z["map_logits"].argmax(0).astype(np.uint8), z["occ"].astype(np.uint8))
        for n, (pm, po) in preds.items():
            a = acc[n]
            cm = counts(pm, gmap, 6)
            a["map"] += cm
            a["map_scene"].setdefault(sc, np.zeros((6, 2)))
            a["map_scene"][sc] += cm
            for k, xm in xmask.items():
                a["map_x"][k] += counts(pm, gmap, 6, xm)
            co = counts(po, gocc, 10, cam)
            a["occ"] += co
            a["occ_scene"].setdefault(sc, np.zeros((10, 2)))
            a["occ_scene"][sc] += co
            a["geo"] += counts((po != OCC_EMPTY).astype(np.uint8), (gocc != OCC_EMPTY).astype(np.uint8), 2, cam)
            if has_teacher and n != "teacher":
                a["map_who"] += who_right(pm, preds["teacher"][0], gmap)
                a["occ_who"] += who_right(po, preds["teacher"][1], gocc, cam)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(tokens)}", flush=True)

    def who(v):
        return {"disagree_pixels": int(v[0]), "student_right": v[1] / max(v[0], 1), "teacher_right": v[2] / max(v[0], 1),
                "neither": 1 - (v[1] + v[2]) / max(v[0], 1)}

    report = {"frames": len(tokens), "split": args.split, "stride": args.stride, "models": {}}
    for n, a in acc.items():
        r = {"map": ious(a["map"], MAP_CLASSES, range(1, 6)),
             "map_by_x": {k: ious(c, MAP_CLASSES, range(1, 6))["miou"] for k, c in a["map_x"].items()},
             "occ": ious(a["occ"], OCC_CLASSES, OCC_EVAL),
             "occ_geometry_iou": a["geo"][1, 0] / a["geo"][1, 1]}
        if has_teacher and n != "teacher":
            r["map_disagreement_vs_teacher"] = who(a["map_who"])
            r["occ_disagreement_vs_teacher"] = who(a["occ_who"])
        report["models"][n] = r
    rng = np.random.default_rng(0)
    for base in ("teacher", args.baseline) if has_teacher else (args.baseline,):
        if base in acc:
            report[f"bootstrap_vs_{base}"] = {
                n: {"map": bootstrap(acc[base]["map_scene"], a["map_scene"], list(range(1, 6)), args.reps, rng),
                    "occ": bootstrap(acc[base]["occ_scene"], a["occ_scene"], list(OCC_EVAL), args.reps, rng)}
                for n, a in acc.items() if n != base}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float))
    print(f"{'model':12s} {'map mIoU':>8s} {'occ mIoU':>8s} {'occ geo':>8s}  map per class (drive line edge cross walk)")
    for n, r in report["models"].items():
        pc = r["map"]["per_class"]
        print(f"{n:12s} {r['map']['miou']:8.3f} {r['occ']['miou']:8.3f} {r['occ_geometry_iou']:8.3f}  "
              + " ".join(f"{pc[c] or 0:.3f}" for c in MAP_CLASSES[1:]))
    print("->", args.output)


if __name__ == "__main__":
    main()
