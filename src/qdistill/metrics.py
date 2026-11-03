"""Simplified BEV metrics shared by the evaluation scripts.

Detection: greedy centre-distance matching (<= 2 m) of predictions sorted by score (>= 0.3), per class.
GT filter as in the nuScenes detection eval: |x|, |y| <= 51.2 m, >= 1 lidar+radar point, mapped label.
Teacher/student predictions are in the lidar frame with z at the box bottom; ``pred_to_ego`` converts
to the packed-GT convention.
"""

from __future__ import annotations

import numpy as np
import torch

from qwen_drive_perception import geometry

SCORE_THR = 0.3
MATCH_DIST = 2.0
DET_RANGE = 51.2
NUM_DET_CLASSES = 7
OCC_CLASSES = tuple(range(9))  # 9 = empty
MAP_CLASSES = tuple(range(1, 6))  # 0 = background


def pred_to_ego(pred: dict, lidar2ego: np.ndarray, score_thr: float = SCORE_THR) -> tuple[np.ndarray, np.ndarray]:
    """Boxes with score >= score_thr in the ego frame, z at the centre, sorted by score."""
    keep = pred["scores"] >= score_thr
    order = np.argsort(-pred["scores"][keep])
    boxes = torch.from_numpy(np.asarray(pred["boxes"])[keep][order]).float()
    ego = geometry.lidar_to_ego_boxes(boxes, torch.from_numpy(np.asarray(lidar2ego)).float()).numpy()
    ego[:, 2] += ego[:, 5] / 2
    return ego, np.asarray(pred["labels"])[keep][order]


def filter_gt(gt: dict) -> tuple[np.ndarray, np.ndarray]:
    keep = (np.abs(gt["boxes"][:, :2]) <= DET_RANGE).all(1) & (gt["num_pts"] > 0) & (gt["labels"] >= 0)
    return gt["boxes"][keep], gt["labels"][keep]


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray) -> list[tuple[int, int]]:
    used = np.zeros(len(gt_xy), dtype=bool)
    pairs = []
    for i, xy in enumerate(pred_xy):
        cand = np.where(~used)[0]
        if len(cand) == 0:
            break
        d = np.linalg.norm(gt_xy[cand] - xy, axis=1)
        j = int(np.argmin(d))
        if d[j] <= MATCH_DIST:
            used[cand[j]] = True
            pairs.append((i, int(cand[j])))
    return pairs


def det_counts(pred_boxes, pred_labels, ref_boxes, ref_labels) -> np.ndarray:
    """Per-class [tp, n_pred, n_ref], shape (NUM_DET_CLASSES, 3)."""
    out = np.zeros((NUM_DET_CLASSES, 3))
    for k in range(NUM_DET_CLASSES):
        pi, ri = np.where(pred_labels == k)[0], np.where(ref_labels == k)[0]
        out[k] = [len(greedy_match(pred_boxes[pi, :2], ref_boxes[ri, :2])), len(pi), len(ri)]
    return out


def prf(counts: np.ndarray) -> dict:
    tp, n_pred, n_ref = counts
    p = tp / max(n_pred, 1)
    r = tp / max(n_ref, 1)
    return {"precision": float(p), "recall": float(r), "f1": float(2 * p * r / max(p + r, 1e-9))}


def iou_counts(pred: np.ndarray, ref: np.ndarray, classes) -> np.ndarray:
    """Per-class [intersection, union]; sum over frames, then ``miou``."""
    return np.array([[np.sum((pred == c) & (ref == c)), np.sum((pred == c) | (ref == c))] for c in classes], dtype=np.float64)


def miou(counts: np.ndarray) -> float:
    valid = counts[:, 1] > 0
    return float(np.mean(counts[valid, 0] / counts[valid, 1]))


def tap_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = torch.from_numpy(np.asarray(a)).float().reshape(-1, a.shape[-1])
    b = torch.from_numpy(np.asarray(b)).float().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
