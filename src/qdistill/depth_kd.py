"""Losses through the frozen teacher head's ViT (geometry) path: head-space features and depth.

The head turns the ViT tap into depth with vit_neck (SimpleFPN, 0.85M) -> DepthNet (ASPP, 5.0M) -> softmax over 118
bins (1 m to 60 m in 0.5 m steps), which feeds the UVTR voxel pooling (the ViT tap drives localisation). This module holds
frozen copies of those two blocks; gradients flow to the student's ViT tap only. Teacher distributions are computed on
the fly from the cached teacher taps (no extra cache).

Head space: the ViT path starts with Conv1x1 1024->256 (no bias) + LayerNorm2d (vit_neck.stages[0][:2]) and the tap is
used nowhere else, so the head is blind to W's 768-dim null space and to per-patch scale after W.

Depth bins: arange(1.0, 60.0, 0.5) (FRUSTUM_RANGE / view_transform.py, camera-frame depth). weights the depth KL
towards far foreground patches via the teacher's expected depth. Adaptor space: the LLM tap is used only by the
head's adaptor (SimpleFPN 2560 -> 4 scales x 256), the analogue of head space for the LLM tap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qdistill.teacher import load_head


class HeadDepth(nn.Module):
    def __init__(self, weights_dir, device: str = "cuda"):
        super().__init__()
        bm = load_head(weights_dir, device).bev_modeling
        self.vit_neck, self.depth_net, self.adaptor = bm.vit_neck, bm.depth_net, bm.adaptor
        self.register_buffer("depth_bins", torch.arange(1.0, 60.0, 0.5, device=device), persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # always eval: frozen BatchNorm statistics
        return super().train(False)

    def head_space(self, vit: torch.Tensor) -> torch.Tensor:
        """Raw ViT tap ``[B, N, H, W, 1024]`` -> what the head sees after Conv1x1 + LayerNorm ``[B, N, H, W, 256]``."""
        b, n = vit.shape[:2]
        x = vit.flatten(0, 1).permute(0, 3, 1, 2).to(next(self.parameters()).dtype)
        stage = self.vit_neck.stages[0]
        y = stage[1](stage[0](x)).float()
        return y.permute(0, 2, 3, 1).reshape(b, n, *y.shape[2:], y.shape[1])

    def forward(self, vit: torch.Tensor) -> torch.Tensor:
        """``vit``: raw (de-normalised) tap ``[B, N, H, W, C]`` -> depth logits ``[B, N, D, H, W]`` (float32)."""
        b, n = vit.shape[:2]
        x = vit.flatten(0, 1).permute(0, 3, 1, 2).to(next(self.parameters()).dtype)
        logits = self.depth_net(self.vit_neck(x)[0])
        return logits.float().view(b, n, *logits.shape[1:])


def expected_depth(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Depth logits ``[B, N, D, H, W]`` -> expected depth in metres ``[B, N, H, W]``."""
    assert logits.shape[2] == bins.numel(), (logits.shape, bins.numel())
    return (F.softmax(logits.float(), dim=2) * bins.view(1, 1, -1, 1, 1)).sum(2)


def range_ramp(depth: torch.Tensor, near: float = 15.0, far: float = 30.0) -> torch.Tensor:
    """0 below ``near`` metres, 1 beyond ``far``, linear in between."""
    return ((depth - near) / (far - near)).clamp(0.0, 1.0)


def adaptor_loss(adaptor: nn.Module, s_llm: torch.Tensor, t_llm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw LLM taps ``[B, N, H, W, 2560]`` -> mean over the adaptor's 4 output scales of MSE / var(teacher) and cosine
    over channels. Gradients reach ``s_llm`` only."""
    def run(x):
        x = x.flatten(0, 1).permute(0, 3, 1, 2).to(next(adaptor.parameters()).dtype)
        return [f.float() for f in adaptor(x)]
    s = run(s_llm)
    with torch.no_grad():
        t = run(t_llm)
    mse = torch.stack([(a - b).pow(2).mean() / b.var() for a, b in zip(s, t)]).mean()
    cos = torch.stack([F.cosine_similarity(a, b, dim=1).mean() for a, b in zip(s, t)]).mean()
    return mse, cos


def head_space_loss(s: torch.Tensor, t: torch.Tensor, weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE and mean cosine between head-space features ``[B, N, H, W, 256]``, per-patch weighted means."""
    mse_tok = (s - t).pow(2).mean(-1)
    cos_tok = F.cosine_similarity(s, t, dim=-1)
    if weight is None:
        return mse_tok.mean(), cos_tok.mean()
    w = weight.float()
    return (w * mse_tok).sum() / w.sum(), (w * cos_tok).sum() / w.sum()


def depth_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """KL(teacher || student) over depth bins per patch, weighted mean over patches. weight: ``[B, N, H, W]``."""
    log_s = F.log_softmax(student_logits, dim=2)
    log_t = F.log_softmax(teacher_logits, dim=2)
    kl = (log_t.exp() * (log_t - log_s)).sum(dim=2)  # [B, N, H, W]
    if weight is None:
        return kl.mean()
    return (weight * kl).sum() / weight.sum()
