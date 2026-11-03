"""GT term: BEVFormer's detection loss against the packed GT boxes (Hungarian matching, per decoder layer).

    from qdistill.det_gt_loss import gt_targets, gt_losses
    gt = gt_targets(packed / token)                         # filtered as metrics.filter_gt, normalised box code
    Lo = gt_losses(s["all_cls_scores"], s["all_bbox_preds"], gt)     # {"gt_cls", "gt_box"}

As upstream BEVFormer (mmdet3d config of the released head family): HungarianAssigner3D with FocalLossCost (weight 2)
+ BBox3DL1Cost on the first 8 code dims (weight 0.25); losses sigmoid focal (alpha 0.25, gamma 2, weight 2) and L1
(weight 0.25, code weights 1 x 8, 0.2 for vx vy), both divided by the number of GT boxes; every decoder layer
(auxiliary losses), averaged over layers. Box code [cx cy log w log l cz log h sin cos vx vy]: the head's decoded
all_bbox_preds (centre in metres, sizes in log) against normalised GT, as BEVFormer's normalize_bbox. GT: ego frame at
the lidar timestamp, w along the heading, z at the box centre.
GT velocity is NaN for annotations without a neighbour (6.5% of s1_C4k[::2] frames, 2026-10-25 smoke): those vx vy are
set to 0 with L1 weight 0 (matching uses the first 8 dims only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from qdistill.metrics import filter_gt

CODE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2)
W_CLS, W_BOX, ALPHA, GAMMA = 2.0, 0.25, 0.25, 2.0


def gt_targets(frame_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(normalised boxes [M, 10] fp32 cuda, labels [M] int64 cuda, L1 weights [M, 10]) of a packed frame."""
    g = np.load(Path(frame_dir) / "gt.npz")
    return encode_gt(*filter_gt({k: g[k] for k in g.files}))


def encode_gt(boxes: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ego-frame GT boxes [M, 9] + labels -> (normalised boxes, labels, L1 weights) on cuda (as gt_targets)."""
    b = torch.as_tensor(boxes, dtype=torch.float32)
    code = torch.stack([b[:, 0], b[:, 1], b[:, 3].clamp_min(1e-3).log(), b[:, 4].clamp_min(1e-3).log(), b[:, 2],
                        b[:, 5].clamp_min(1e-3).log(), b[:, 6].sin(), b[:, 6].cos(), b[:, 7], b[:, 8]], -1)
    w = torch.tensor(CODE_WEIGHTS).expand_as(code).clone()
    w[~torch.isfinite(code)] = 0.0
    code = torch.nan_to_num(code, nan=0.0, posinf=0.0, neginf=0.0)
    return code.cuda(), torch.as_tensor(labels, dtype=torch.int64).cuda(), w.cuda()


def _match(cls: torch.Tensor, box: torch.Tensor, gt_box: torch.Tensor, gt_lab: torch.Tensor):
    with torch.no_grad():
        p = cls.float().sigmoid()
        neg = -(1 - p + 1e-12).log() * (1 - ALPHA) * p.pow(GAMMA)
        pos = -(p + 1e-12).log() * ALPHA * (1 - p).pow(GAMMA)
        cost = W_CLS * (pos[:, gt_lab] - neg[:, gt_lab]) + W_BOX * torch.cdist(box[:, :8].float(), gt_box[:, :8], p=1)
        q, g = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(q, device=cls.device), torch.as_tensor(g, device=cls.device)


def _layer(cls, box, gt_box, gt_lab, gt_w):
    n_gt = max(len(gt_lab), 1)
    target = torch.zeros_like(cls, dtype=torch.float32)
    lb = cls.new_zeros(())
    if len(gt_lab):
        q, g = _match(cls, box, gt_box, gt_lab)
        target[q, gt_lab[g]] = 1.0
        lb = ((box[q].float() - gt_box[g]).abs() * gt_w[g]).sum() / n_gt
    lc = _focal(cls.float(), target).sum() / n_gt
    return W_CLS * lc, W_BOX * lb


def _focal(x, t):
    """Sigmoid focal loss per element (torchvision.ops.sigmoid_focal_loss, reduction none)."""
    p = x.sigmoid()
    ce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    return ce * (1 - pt).pow(GAMMA) * (ALPHA * t + (1 - ALPHA) * (1 - t))


def gt_losses(all_cls: torch.Tensor, all_box: torch.Tensor, gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict:
    """all_cls [L, 1, Q, 7], all_box [L, 1, Q, 10] (head outputs) -> {"gt_cls", "gt_box"} averaged over the L layers."""
    gt_box, gt_lab, gt_w = gt
    parts = [_layer(all_cls[i, 0], all_box[i, 0], gt_box, gt_lab, gt_w) for i in range(all_cls.shape[0])]
    return {"gt_cls": sum(c for c, _ in parts) / len(parts), "gt_box": sum(b for _, b in parts) / len(parts)}
