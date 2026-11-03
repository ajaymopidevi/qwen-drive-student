"""Student loading and the head module table used by the timing scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from qdistill.student import TapNormalizer, TapStudent

MODULES = {  # report name: attribute path under head.bev_modeling
    "llm_adaptor (FPN)": "adaptor",
    "vit_neck": "vit_neck",
    "depth_net": "depth_net",
    "voxel_pool (view_trans)": "view_trans",
    "uvtr_query_proj": "uvtr_query_proj",
    "bev_encoder": "head.transformer.encoder",
    "det_decoder": "head.transformer.decoder",
    "occ_fuse (proj+fuse)": ("head.transformer.uvtr_occ_proj", "head.transformer.uvtr_occ_fuse"),
    "occ_refiner (U-Net)": "head.transformer.occ_decoder",
    "occ_pred_head": "head.transformer.occ_pred_head",
    "map_seg (U-Net)": "head.seg_decoder",
}


def get(mod: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def load_student(ckpt_path: Path, cache: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    stats = a.get("tap_stats") or cache / f"tap_stats_{a['train_split'].replace(':', '_')}.npz"
    student = TapStudent(a["backbone"], pretrained=False).cuda().eval()
    student.load_state_dict(ckpt["model"])
    return student, TapNormalizer(dict(np.load(stats))).cuda()
