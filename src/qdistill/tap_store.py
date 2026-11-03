"""Compressed teacher-tap storage: what is written to disk and how training reads it back.

Format "vitW+llmInt8" (one .npz per frame, 12.4 MB instead of 35.8):
  vitW       [N, 32, 56, 256] fp16   W x, W = the head's ViT Conv1x1 1024->256 (vit_neck.stages[0][0], no bias)
  llm_q      [N, 16, 28, 2560] int8  per-frame per-channel symmetric int8 of the LLM tap
  llm_scale  [2560] fp16             its scales
Decoding gives the ViT tap as pinv(W) W x: the head sees exactly W x, the
768-dim null space of W is zero. W is saved once per cache as <cache>/vit_W.npy (convert_tap_store.py). Same
arithmetic as the stored formats (vitW, llmInt8), so the head checks apply to these files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

FORMATS = ("full", "vitW", "llmInt8", "vitW+llmInt8")


def int8_encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric int8 over all tokens of the frame; scale stored in fp16."""
    xf = x.float().reshape(-1, x.shape[-1])
    scale = (xf.abs().amax(0) / 127.0).clamp_min(1e-8).half()
    q = torch.clamp(torch.round(xf / scale.float()), -127, 127).to(torch.int8)
    return q.reshape(x.shape), scale


def int8_decode(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


def encode(fmt: str, vit: torch.Tensor, llm: torch.Tensor, W: torch.Tensor) -> dict[str, np.ndarray]:
    """Taps (any float dtype) -> arrays to store. W: [256, 1024] float."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown tap format {fmt!r}, expected one of {FORMATS}")
    parts = fmt.split("+")
    out = {}
    if "vitW" in parts:
        out["vitW"] = (vit.float() @ W.T.to(vit.device)).half()
    else:
        out["vit"] = vit.half()
    if "llmInt8" in parts:
        out["llm_q"], out["llm_scale"] = int8_encode(llm)
    else:
        out["llm"] = llm.half()
    return {k: v.cpu().numpy() for k, v in out.items()}


def decode(arrays, W_pinv: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stored arrays (npz or dict, any format above) -> float32 (vit, llm) taps. W_pinv: [1024, 256], needed for vitW."""
    keys = set(arrays.keys()) if hasattr(arrays, "keys") else set(arrays.files)
    if "vitW" in keys:
        if W_pinv is None:
            raise ValueError("vitW-format taps need W_pinv (<cache>/vit_W.npy)")
        vit = torch.from_numpy(arrays["vitW"]).float() @ W_pinv.T
    else:
        vit = torch.from_numpy(arrays["vit"]).float()
    if "llm_q" in keys:
        llm = int8_decode(torch.from_numpy(arrays["llm_q"]), torch.from_numpy(arrays["llm_scale"]))
    else:
        llm = torch.from_numpy(arrays["llm"]).float()
    return vit, llm


def load_W(cache_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """(W [256, 1024], pinv(W) [1024, 256]) saved next to a converted cache."""
    W = torch.from_numpy(np.load(Path(cache_root) / "vit_W.npy")).float()
    return W, torch.linalg.pinv(W)


def head_W(weights_dir) -> torch.Tensor:
    """The head's ViT Conv1x1 weight as [256, 1024] float32 (loads the head on CPU)."""
    from qdistill.teacher import load_head

    head = load_head(weights_dir, device="cpu")
    W = head.bev_modeling.vit_neck.stages[0][0].weight.detach().float().squeeze(-1).squeeze(-1).clone()
    del head
    return W
