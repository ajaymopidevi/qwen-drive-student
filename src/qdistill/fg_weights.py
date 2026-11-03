"""Foreground maps on the teacher's ViT patch grid from projected GT boxes.

Each packed GT box (>= 1 lidar+radar point, mapped label) is converted to the lidar frame, its 8 corners are projected
into each camera at the teacher's 896x512 input (the head's own lidar2img), and the 2D box hull marks the 16x16-px
patches of the 32x56 ViT grid it covers. A patch takes the largest class weight of the boxes covering it; background 0.
Boxes with a corner behind the camera are skipped (geometry.project_to_image), as in the teacher's visualiser.
"""

from __future__ import annotations

import numpy as np
import torch

from qwen_drive_perception import geometry
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES

GRID = (32, 56)  # (H, W) of the ViT tap; 16 px patches at 512x896
IMAGE_SIZE = (896, 512)  # (W, H)
DEFAULT_CLASS_WEIGHTS = {"vehicle": 1.0, "czone_sign": 1.0, "bicycle": 3.0, "generic_object": 2.0,
                         "pedestrian": 3.0, "traffic_cone": 1.0, "barrier": 1.0}


def fg_class_map(frame, class_weights: dict[str, float] = DEFAULT_CLASS_WEIGHTS) -> np.ndarray:
    """``[N_cam, 32, 56]`` float32: max class weight of the GT boxes covering each patch (0 = background)."""
    n_cam = len(frame.cam_order)
    out = np.zeros((n_cam, *GRID), dtype=np.float32)
    gt = frame.gt
    keep = (gt["num_pts"] > 0) & (gt["labels"] >= 0)
    if not keep.any():
        return out
    boxes_ego = torch.from_numpy(np.asarray(gt["boxes"][keep], dtype=np.float32))
    boxes = geometry.ego_to_lidar_boxes(boxes_ego, torch.from_numpy(np.asarray(frame.lidar2ego, dtype=np.float32))).numpy()
    boxes[:, 2] -= boxes[:, 5] / 2  # packed GT has z at the centre; box_corners wants the bottom
    corners = geometry.box_corners(boxes[:, :7])
    weights = np.array([class_weights[DET_CLASS_NAMES[k]] for k in gt["labels"][keep]], dtype=np.float32)
    lidar2img = frame.img_metas(image_size=IMAGE_SIZE)["lidar2img"]
    W, H = IMAGE_SIZE
    ph, pw = H / GRID[0], W / GRID[1]
    for c in range(n_cam):
        uv, valid = geometry.project_to_image(corners, lidar2img[c], W, H)
        for i in np.nonzero(valid)[0]:
            u0, v0 = np.clip(uv[i].min(0), 0, [W - 1, H - 1])
            u1, v1 = np.clip(uv[i].max(0), 0, [W - 1, H - 1])
            r0, r1 = int(v0 // ph), int(v1 // ph)
            c0, c1 = int(u0 // pw), int(u1 // pw)
            region = out[c, r0 : r1 + 1, c0 : c1 + 1]
            np.maximum(region, weights[i], out=region)
    return out


def gt_depth_map(frame) -> np.ndarray:
    """``[N_cam, 32, 56]`` float32: camera depth (m) of the nearest GT box centre whose 2D hull covers each patch, 0 =
    background. Same boxes and hulls as ``fg_class_map``; used for range weighting, since the teacher's depth
    softmax is not metric (most fg mass sits in out-of-grid bins)."""
    n_cam = len(frame.cam_order)
    out = np.full((n_cam, *GRID), np.inf, dtype=np.float32)
    gt = frame.gt
    keep = (gt["num_pts"] > 0) & (gt["labels"] >= 0)
    if keep.any():
        boxes_ego = torch.from_numpy(np.asarray(gt["boxes"][keep], dtype=np.float32))
        boxes = geometry.ego_to_lidar_boxes(boxes_ego, torch.from_numpy(np.asarray(frame.lidar2ego, dtype=np.float32))).numpy()
        centre = np.c_[boxes[:, :3].astype(np.float64), np.ones(len(boxes))]  # packed z is the box centre
        boxes[:, 2] -= boxes[:, 5] / 2
        corners = geometry.box_corners(boxes[:, :7])
        lidar2img = frame.img_metas(image_size=IMAGE_SIZE)["lidar2img"]
        W, H = IMAGE_SIZE
        ph, pw = H / GRID[0], W / GRID[1]
        for c in range(n_cam):
            uv, valid = geometry.project_to_image(corners, lidar2img[c], W, H)
            z = (np.asarray(lidar2img[c], dtype=np.float64) @ centre.T)[2]
            for i in np.nonzero(valid)[0]:
                u0, v0 = np.clip(uv[i].min(0), 0, [W - 1, H - 1])
                u1, v1 = np.clip(uv[i].max(0), 0, [W - 1, H - 1])
                region = out[c, int(v0 // ph) : int(v1 // ph) + 1, int(u0 // pw) : int(u1 // pw) + 1]
                np.minimum(region, z[i], out=region)
    out[np.isinf(out)] = 0.0
    return out
