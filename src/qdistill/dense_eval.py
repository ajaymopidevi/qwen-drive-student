"""in-process dense GT scoring (map mIoU vs map_gt rasters, occupancy mIoU vs Occ3D on mask_camera), the same
protocol as scripts/evaluation/eval_dense_gt.py but without dumps, plus per-scene counts for a paired scene bootstrap.

    sc = DenseScorer()
    sc.add(scene, map_pred [200, 400] uint8, occ10 [200, 200, 16] uint8 teacher classes, gt, occ18=None)
    sc.summary() -> {"map": miou, "occ": miou, ...};  sc.save(path)

Occupancy predictions of an 18-class (Occ3D) head are scored two ways (``occ_rules``):
  argmax  argmax over the 18 Occ3D classes, then OCC3D_TO_TEACHER (as the GT is mapped)
  merge   logsumexp of each teacher class's member logits (czone_sign has none: never predicted), then argmax
and additionally as the standard Occ3D mIoU over classes 0..16 (free = 17 excluded) on mask_camera ("occ3d").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np



MAP_CLASSES = ("background", "driveable_surface", "road_line", "road_edge", "crosswalk", "walkway")
OCC_CLASSES = ("vehicle", "czone_sign", "bicycle", "generic_object", "pedestrian", "traffic_cone", "barrier",
               "driveable", "background", "empty")
OCC_EVAL = (0, 2, 3, 4, 5, 6, 7, 8)  # teacher classes with Occ3D GT (eval_dense_gt.OCC_EVAL)
MAP_EVAL = (1, 2, 3, 4, 5)
OCC3D_EVAL = tuple(range(17))
OCC3D_TO_TEACHER = (3, 6, 2, 0, 0, 0, 2, 4, 5, 0, 0, 7, 8, 8, 8, 8, 8, 9)  # Occ3D class -> teacher occupancy class
O2T = np.asarray(OCC3D_TO_TEACHER, np.uint8)
ROOT = Path(__file__).resolve().parents[2]


def load_gt(token: str, scene: str, map_gt=ROOT / "data/gt/map", occ_gt=ROOT / "data/occ3D_nuScenes/gts") -> dict:
    """{"map": [200, 400] uint8, "occ18": [200, 200, 16] uint8 Occ3D semantics, "occ10": teacher classes,
    "cam": [200, 200, 16] bool mask_camera}."""
    o3 = np.load(Path(occ_gt) / scene / token / "labels.npz")
    sem = o3["semantics"]
    return {"map": np.load(Path(map_gt) / f"{token}.npy"), "occ18": sem, "occ10": O2T[sem],
            "cam": o3["mask_camera"].astype(bool)}




def counts(pred: np.ndarray, gt: np.ndarray, n: int, mask=None) -> np.ndarray:
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    c = np.bincount(gt.ravel().astype(np.int64) * n + pred.ravel().astype(np.int64), minlength=n * n).reshape(n, n)
    inter = np.diag(c)
    return np.stack([inter, c.sum(0) + c.sum(1) - inter], 1).astype(np.float64)


def miou(c: np.ndarray, keep) -> float:
    v = [c[k, 0] / c[k, 1] for k in keep if c[k, 1]]
    return float(np.mean(v)) if v else float("nan")


class DenseScorer:
    def __init__(self):
        self.scene: dict[str, dict[str, np.ndarray]] = {}
        self.frames = 0

    def _acc(self, scene, key, c):
        d = self.scene.setdefault(scene, {})
        d[key] = d.get(key, 0) + c

    def add(self, scene: str, map_pred: np.ndarray, occ10: dict | np.ndarray, gt: dict, occ18: np.ndarray | None = None):
        """``occ10``: one teacher-class grid, or {rule: grid} for several decoding rules."""
        self.frames += 1
        self._acc(scene, "map", counts(map_pred, gt["map"], 6))
        for rule, g in (occ10.items() if isinstance(occ10, dict) else (("native", occ10),)):
            self._acc(scene, f"occ_{rule}", counts(g, gt["occ10"], 10, gt["cam"]))
            self._acc(scene, f"geo_{rule}", counts((g != 9).astype(np.uint8), (gt["occ10"] != 9).astype(np.uint8), 2,
                                                    gt["cam"]))
        if occ18 is not None:
            self._acc(scene, "occ3d", counts(occ18, gt["occ18"], 18, gt["cam"]))

    def totals(self) -> dict:
        tot = {}
        for d in self.scene.values():
            for k, v in d.items():
                tot[k] = tot.get(k, 0) + v
        return tot

    def summary(self) -> dict:
        out = {"frames": self.frames}
        for k, c in self.totals().items():
            if k == "map":
                out["map"] = round(miou(c, MAP_EVAL), 5)
                out["map_per_class"] = {MAP_CLASSES[i]: round(c[i, 0] / max(c[i, 1], 1), 4) for i in MAP_EVAL}
            elif k.startswith("occ_"):
                out[k] = round(miou(c, OCC_EVAL), 5)
            elif k.startswith("geo_"):
                out[k] = round(c[1, 0] / max(c[1, 1], 1), 5)
            elif k == "occ3d":
                out["occ3d_17"] = round(miou(c, OCC3D_EVAL), 5)
        return out

    def save(self, path, extra: dict | None = None):
        Path(path).write_text(json.dumps({"summary": self.summary(), **(extra or {}),
                                          "scenes": {s: {k: v.tolist() for k, v in d.items()} for s, d in self.scene.items()}}))


def load_scenes(path) -> dict:
    d = json.loads(Path(path).read_text())
    return {s: {k: np.asarray(v) for k, v in dd.items()} for s, dd in d["scenes"].items()}
