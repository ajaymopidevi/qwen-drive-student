"""Stage-1 student: a small image backbone that regresses the teacher's two VLM taps.

    images [B, N_cam, 3, 512, 896]
      -> HGNetV2 (timm) stride-16 / stride-32 features
      -> vit branch: FPN-lite at stride 16 -> 1024 ch        ~ teacher ViT tap  [B, N, 32, 56, 1024]
      -> llm branch: stride-32 tokens of *all* cameras -> transformer encoder -> 2560 ch
                                                           ~ teacher LLM tap  [B, N, 16, 28, 2560]

The LLM tap is contextual (the VLM attends across every camera and the prompt), so the llm branch
mixes all cameras' tokens with a transformer instead of predicting each image independently.
Outputs are standardised taps ((x - mean) / std with per-channel train-split statistics); use
``denormalize`` before feeding the frozen teacher head.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

VIT_DIM, LLM_DIM = 1024, 2560
LLM_GRID = (16, 28)
MAX_CAMS = 8


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class TapStudent(nn.Module):
    def __init__(
        self,
        backbone: str = "hgnetv2_b4.ssld_stage2_ft_in1k",
        pretrained: bool = True,
        neck_dim: int = 256,
        llm_model_dim: int = 384,
        llm_layers: int = 4,
        llm_heads: int = 8,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(2, 3))
        c16, c32 = self.backbone.feature_info.channels()
        self.lat16 = ConvBNAct(c16, neck_dim)
        self.lat32 = ConvBNAct(c32, neck_dim)

        self.vit_neck = nn.Sequential(ConvBNAct(neck_dim, neck_dim, 3), ConvBNAct(neck_dim, neck_dim, 3))
        self.vit_out = nn.Conv2d(neck_dim, VIT_DIM, 1)

        self.llm_in = nn.Conv2d(2 * neck_dim, llm_model_dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, LLM_GRID[0] * LLM_GRID[1], llm_model_dim))
        self.cam_embed = nn.Embedding(MAX_CAMS, llm_model_dim)
        layer = nn.TransformerEncoderLayer(
            llm_model_dim, llm_heads, dim_feedforward=4 * llm_model_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, llm_layers, enable_nested_tensor=False)
        self.llm_out = nn.Sequential(nn.LayerNorm(llm_model_dim), nn.Linear(llm_model_dim, LLM_DIM))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cam_embed.weight, std=0.02)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.backbone.set_grad_checkpointing(enable)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = images.shape[:2]
        c16, c32 = self.backbone(images.flatten(0, 1))
        p32 = self.lat32(c32)
        p16 = self.lat16(c16) + F.interpolate(p32, size=c16.shape[-2:], mode="nearest")

        vit = self.vit_out(self.vit_neck(p16))  # [BN, 1024, 32, 56]
        vit = vit.permute(0, 2, 3, 1).reshape(B, N, *vit.shape[-2:], VIT_DIM)

        tok = self.llm_in(torch.cat([p32, F.avg_pool2d(p16, 2)], dim=1))  # [BN, d, 16, 28]
        h, w = tok.shape[-2:]
        assert (h, w) == LLM_GRID, f"expected a {LLM_GRID} stride-32 grid (896x512 input), got {(h, w)}"
        tok = tok.flatten(2).transpose(1, 2) + self.pos_embed  # [BN, hw, d]
        tok = tok.view(B, N, h * w, -1) + self.cam_embed.weight[:N, None, :]
        tok = self.encoder(tok.reshape(B, N * h * w, -1))
        llm = self.llm_out(tok).view(B, N, h, w, LLM_DIM)
        return vit, llm


class TapNormalizer(nn.Module):
    """Per-channel standardisation of the taps with train-split statistics (scripts/backbone/tap_stats.py)."""

    def __init__(self, stats: dict):
        super().__init__()
        for name in ("vit", "llm"):
            self.register_buffer(f"{name}_mean", torch.as_tensor(stats[f"{name}_mean"]).float())
            self.register_buffer(f"{name}_std", torch.as_tensor(stats[f"{name}_std"]).float())

    def normalize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std")

    def denormalize(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return x.float() * getattr(self, f"{name}_std") + getattr(self, f"{name}_mean")


def tap_loss(pred: dict, target_norm: dict, norm: TapNormalizer, cos_weight: float = 1.0,
             vit_weight: torch.Tensor | None = None, vit_scale: float = 1.0) -> dict:
    """MSE on standardised taps + (1 - cosine) on raw taps, which is the space the head sees.

    ``vit_weight`` ``[B, N, H, W]``: per-patch weights for the ViT terms (weighted means); LLM terms unweighted.
    ``vit_scale`` scales the ViT terms in the total (0 = raw ViT tap unconstrained, e.g. with a head-space loss).
    """
    out = {}
    total = 0.0
    for name in ("vit", "llm"):
        p, t = pred[name].float(), target_norm[name].float()
        cos_tok = F.cosine_similarity(norm.denormalize(name, p), norm.denormalize(name, t), dim=-1)
        if name == "vit" and vit_weight is not None:
            w = vit_weight.float()
            mse = (w * (p - t).pow(2).mean(-1)).sum() / w.sum()
            cos = (w * cos_tok).sum() / w.sum()
        else:
            mse = F.mse_loss(p, t)
            cos = cos_tok.mean()
        out[f"{name}_mse"] = mse.detach()
        out[f"{name}_cos"] = cos.detach()
        total = total + (vit_scale if name == "vit" else 1.0) * (mse + cos_weight * (1 - cos))
    out["loss"] = total
    return out
