"""Foreground mask on the voxel block's BEV grid from GT boxes.

The head's voxel grid lives in the ego frame (view_transform.coord_preparing maps frustum points through lidar2ego
before indexing: index = (p - pc_range) / voxel_size), and packed GT boxes are ego-frame, so boxes map onto it
directly. A cell is foreground when its centre lies inside a GT box (>= 1 lidar+radar point, mapped label) whose
footprint is widened by ``margin`` metres on each side. Output rows are Y, columns X: the [H, W] = [Y, X] layout of
the block's output [B, C, Z, Y, X] and of uvtr_query_proj's BEV map.
"""

from __future__ import annotations

import numpy as np

from qwen_drive_perception import geometry


def bev_fg_mask(gt: dict, pc_range, voxel_size, voxel_shape, margin: float = 1.0, labels=None) -> np.ndarray:
    """``[Y, X]`` bool: BEV cells covered by a (margin-widened) GT box footprint; ``labels``: only these label ids."""
    X, Y = int(voxel_shape[0]), int(voxel_shape[1])
    mask = np.zeros((Y, X), dtype=bool)
    lab = np.asarray(gt["labels"])
    keep = (np.asarray(gt["num_pts"]) > 0) & (lab >= 0)
    if labels is not None:
        keep &= np.isin(lab, list(labels))
    if not keep.any():
        return mask
    boxes = np.asarray(gt["boxes"], dtype=np.float64)[keep, :7].copy()
    boxes[:, 3:5] += 2 * margin
    quads = geometry.box_corners(boxes)[:, :4, :2]  # bottom face
    xs = pc_range[0] + (np.arange(X) + 0.5) * voxel_size[0]
    ys = pc_range[1] + (np.arange(Y) + 0.5) * voxel_size[1]
    for q in quads:
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        ix = np.nonzero((xs >= x0) & (xs <= x1))[0]
        iy = np.nonzero((ys >= y0) & (ys <= y1))[0]
        if not len(ix) or not len(iy):
            continue
        px, py = np.meshgrid(xs[ix], ys[iy])  # [len(iy), len(ix)]
        pos = np.ones(px.shape, dtype=bool)
        neg = np.ones(px.shape, dtype=bool)
        for k in range(4):  # convex quad: inside = same side of every edge (winding-independent; box_corners'
            ax, ay = q[k]   # order is clockwise in x-right / y-up despite its docstring)
            bx, by = q[(k + 1) % 4]
            cr = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            pos &= cr >= 0
            neg &= cr <= 0
        mask[np.ix_(iy, ix)] |= pos | neg
    return mask
