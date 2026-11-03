"""Stage 3 Phase B: a FlashOcc-width 2D occupancy refiner in place of the 3D U-Net.

The critique showed a stage-matched 2D refiner with Z folded into
channels costs ~5x the 3D one's FLOPs, so the trunk is narrow (w / 2w / 4w at 200 / 100 / 50) and Z appears only at
the input fold and the output unfold. (TensorRT fp16, random weights): w64 2.7 ms, w128 7.0 ms, w256 23.4 ms
vs the deployed w1 3D refiner 57.6 ms.

Same I/O as the upstream OccVoxelUNetRefiner / NarrowRefiner: [B, 16, Z=16, Y, X] -> [B, 32, Z, Y, X], so its engine
drops into TrtHead / eval_stage1 --trt-refiner and the frozen occ_pred_head runs unchanged. Channels fold as c * Z + z.
No teacher init (no tensor shape matches); trained by distillation from the deployed refiner (distill_refiner.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdistill.bev2d import BasicBlock, conv_bn

CONFIGS = {
    "occ2d_w64": dict(kind="occ2d", w=64),
    "occ2d_w128": dict(kind="occ2d", w=128),
    "occ2d_w256": dict(kind="occ2d", w=256),
}


class Occ2DRefiner(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 32, z: int = 16, w: int = 128, kind: str = "occ2d"):
        super().__init__()
        self.cfg = {"kind": kind, "in_channels": in_channels, "out_channels": out_channels, "z": z, "w": w}
        self.z, self.cout = z, out_channels
        cin = in_channels * z
        self.inp = nn.Sequential(conv_bn(cin, w, 3), BasicBlock(w, w))
        self.e1 = nn.Sequential(BasicBlock(w, 2 * w, 2), BasicBlock(2 * w, 2 * w))
        self.e2 = nn.Sequential(BasicBlock(2 * w, 4 * w, 2), BasicBlock(4 * w, 4 * w))
        self.d1 = nn.Sequential(conv_bn(4 * w + 2 * w, 2 * w, 3), BasicBlock(2 * w, 2 * w))
        self.d0 = nn.Sequential(conv_bn(2 * w + w, w, 3), BasicBlock(w, w))
        self.out = nn.Conv2d(w, out_channels * z, 1)  # the 3D refiner ends in BN + ReLU: outputs are >= 0

    def forward(self, x):
        b, c, z, h, w = x.shape
        x0 = self.inp(x.reshape(b, c * z, h, w))
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        y = self.d1(torch.cat([F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1], 1))
        y = self.d0(torch.cat([F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False), x0], 1))
        return F.relu(self.out(y)).reshape(b, self.cout, z, h, w)


def build(name_or_cfg) -> Occ2DRefiner:
    cfg = CONFIGS[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    return Occ2DRefiner(**cfg)
