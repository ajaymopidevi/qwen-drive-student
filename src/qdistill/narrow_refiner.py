"""Stage 2: the upstream occupancy refiner's U-Net with narrower / shallower high-resolution stages.

the full / half-resolution stages (input_proj, enc1, dec0, dec1, out_block)
take 59% of the refiner's 120 ms with 5.7M of its 52.5M parameters; the deep stages hold the parameters and little of
the time. So the student keeps the upstream structure (qwen_drive_perception.occ_refiner building blocks, same stage
names, so a teacher state dict maps 1:1) with configurable channels c0 / c1 at full / half resolution and blocks per
stage. The deep stages (enc2 / enc3 / bottleneck / dec2) keep the teacher's widths, so their weights copy over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from qwen_drive_perception.occ_refiner import ResidualStage3D

STAGES = ("input_proj", "enc1", "enc2", "enc3", "bottleneck", "dec2", "dec1", "dec0", "out_block")

# name -> cfg; "exact" is the upstream OccVoxelUNetRefiner(16, 32)
CONFIGS = {
    "exact": dict(ch=(64, 128, 256, 384), blocks=dict()),
    "c32": dict(ch=(32, 128, 256, 384), blocks=dict()),
    "c32_64": dict(ch=(32, 64, 256, 384), blocks=dict()),
    "c32_64_b1": dict(ch=(32, 64, 256, 384), blocks=dict(input_proj=1, dec0=1, out_block=1)),
    "c32_64_b1h": dict(ch=(32, 64, 256, 384), blocks=dict(input_proj=1, dec0=1, out_block=1, enc1=1, dec1=1)),
    "c32_b1": dict(ch=(32, 128, 256, 384), blocks=dict(input_proj=1, dec0=1, out_block=1)),
    "c16_64_b1": dict(ch=(16, 64, 256, 384), blocks=dict(input_proj=1, dec0=1, out_block=1)),
}


class NarrowRefiner(nn.Module):
    """Upstream OccVoxelUNetRefiner (XY-only downsampling, trilinear upsampling, skip concat) with channels ``ch``
    (c0 full res, c1 half, c2 quarter, c3 eighth + bottleneck) and ``blocks`` per stage (default 2)."""

    def __init__(self, in_channels: int = 16, out_channels: int = 32, ch=(64, 128, 256, 384), blocks: dict | None = None):
        super().__init__()
        c0, c1, c2, c3 = ch
        nb = {s: 2 for s in STAGES}
        nb.update(blocks or {})
        self.cfg = {"in_channels": in_channels, "out_channels": out_channels, "ch": list(ch), "blocks": dict(blocks or {})}
        self.input_proj = ResidualStage3D(in_channels, c0, num_blocks=nb["input_proj"])
        self.enc1 = ResidualStage3D(c0, c1, stride=(1, 2, 2), num_blocks=nb["enc1"])
        self.enc2 = ResidualStage3D(c1, c2, stride=(1, 2, 2), num_blocks=nb["enc2"])
        self.enc3 = ResidualStage3D(c2, c3, stride=(1, 2, 2), num_blocks=nb["enc3"])
        self.bottleneck = ResidualStage3D(c3, c3, num_blocks=nb["bottleneck"])
        self.dec2 = ResidualStage3D(c3 + c2, c2, num_blocks=nb["dec2"])
        self.dec1 = ResidualStage3D(c2 + c1, c1, num_blocks=nb["dec1"])
        self.dec0 = ResidualStage3D(c1 + c0, c0, num_blocks=nb["dec0"])
        self.out_block = ResidualStage3D(c0, c0, num_blocks=nb["out_block"])
        self.out_proj = nn.Sequential(
            nn.Conv3d(c0, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up(x, target):
        return nn.functional.interpolate(x, size=target.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x):
        skip0 = self.input_proj(x)
        skip1 = self.enc1(skip0)
        skip2 = self.enc2(skip1)
        x = self.bottleneck(self.enc3(skip2))
        x = self.dec2(torch.cat([self._up(x, skip2), skip2], dim=1))
        x = self.dec1(torch.cat([self._up(x, skip1), skip1], dim=1))
        x = self.dec0(torch.cat([self._up(x, skip0), skip0], dim=1))
        return self.out_proj(self.out_block(x))


def build(name_or_cfg) -> NarrowRefiner:
    cfg = CONFIGS[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    return NarrowRefiner(**{k: v for k, v in cfg.items() if k in ("in_channels", "out_channels", "ch", "blocks")})


def init_from_teacher(student: nn.Module, teacher: nn.Module) -> dict:
    """Copy every teacher tensor (parameters and BatchNorm buffers) whose name and shape match; the rest keep their
    random init. Returns counts (tensors / parameters copied vs total)."""
    t = teacher.state_dict()
    s = student.state_dict()
    copied = {k: v for k, v in t.items() if k in s and s[k].shape == v.shape}
    s.update({k: v.to(s[k].dtype) for k, v in copied.items()})
    student.load_state_dict(s)
    pnames = {n for n, _ in student.named_parameters()}
    p_total = sum(p.numel() for p in student.parameters())
    p_copied = sum(v.numel() for k, v in copied.items() if k in pnames)
    return {"tensors_copied": len(copied), "tensors_total": len(s), "params_copied_M": round(p_copied / 1e6, 2),
            "params_total_M": round(p_total / 1e6, 2)}
