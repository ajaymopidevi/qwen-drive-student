"""glue removed from the head; wiring only, no engine rebuilt.

last_out(head)   head_chain "last_out", FIRST item (det_sample wraps the head's forward). upstream's forward runs the
                 cls / reg branches and the box decoding on all 6 decoder layers; inference reads only [-1]
                 (fast_postproc). Also caches bev_pos: positional_encoding of an all-zero mask, constant per grid.
                 Outputs keep upstream's indexing ([-1][0]) with a leading dimension of 1. Exact.
vol_fp16(head)   head_chain "vol_fp16", after dense_trt. After the block (fp16 output) the Z8 volume went fp16 -> bf16 ->
                 fp32 -> fp16 for the one 1x1 uvtr_query_proj fold, then bf16 -> fp16 into the dense engine. Now both read
                 the engine's fp16 buffer. The buffer is reused next frame, and nothing reads the volume after the
                 dense engine. Skips one bf16 rounding of the volume, so it is not bit-identical.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def last_out(head) -> dict:
    from qwen_drive_perception.layers import inverse_sigmoid

    hd = head.bev_modeling.head
    pos_cache = {}

    def forward(mlvl_feats, img_metas, prev_bev=None, vit_bev_feat=None, uvtr_bev_feat=None, uvtr_occ_feat=None):
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        object_query_embeds = hd.query_embedding.weight.to(dtype)
        bev_queries = hd.bev_embedding.weight.to(dtype)
        if uvtr_bev_feat is not None:
            bev_queries = bev_queries + uvtr_bev_feat.to(dtype)
        elif vit_bev_feat is not None:
            bev_queries = bev_queries + vit_bev_feat.to(dtype)
        key = (bs, hd.bev_h, hd.bev_w, dtype)
        if key not in pos_cache:
            mask = torch.zeros((bs, hd.bev_h, hd.bev_w), device=bev_queries.device).to(dtype)
            pos_cache[key] = hd.positional_encoding(mask).to(dtype)
        outputs = hd.transformer(
            mlvl_feats, bev_queries, object_query_embeds, hd.bev_h, hd.bev_w,
            grid_length=(hd.real_h / hd.bev_h, hd.real_w / hd.bev_w), bev_pos=pos_cache[key],
            reg_branches=hd.reg_branches, cls_branches=None, img_metas=img_metas, prev_bev=prev_bev,
            uvtr_occ_feat=uvtr_occ_feat)
        bev_embed, hs, init_reference, inter_references, occ_pred, bev_ego_feat_2d = outputs
        lvl = hs.shape[0] - 1
        h = hs[lvl].permute(1, 0, 2)
        reference = inverse_sigmoid(inter_references[lvl - 1] if lvl else init_reference)
        cls = hd.cls_branches[lvl](h)
        tmp = hd.reg_branches[lvl](h)
        tmp[..., 0:2] += reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] += reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        tmp[..., 0:1] = tmp[..., 0:1] * (hd.pc_range[3] - hd.pc_range[0]) + hd.pc_range[0]
        tmp[..., 1:2] = tmp[..., 1:2] * (hd.pc_range[4] - hd.pc_range[1]) + hd.pc_range[1]
        tmp[..., 4:5] = tmp[..., 4:5] * (hd.pc_range[5] - hd.pc_range[2]) + hd.pc_range[2]
        outs = {"bev_embed": bev_embed, "all_cls_scores": cls[None], "all_bbox_preds": tmp[None], "occ_pred": occ_pred}
        return hd._append_map_preds(outs, bs, bev_ego_feat_2d)

    hd.forward = forward
    return {"last_out": "last decoder layer only, bev_pos cached"}


class _Fp16Proj(nn.Module):
    """The folded query projection as an fp16 conv on the block's fp16 buffer; output in the head's bf16."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = copy.deepcopy(conv).half().eval()

    def forward(self, x):
        assert x.dtype == torch.float16, x.dtype
        return self.conv(x).to(torch.bfloat16)


def vol_fp16(head) -> dict:
    bev = head.bev_modeling
    assert hasattr(bev, "_dense_engine"), "vol_fp16: after dense_trt (the engine must be the volume's only other reader)"
    fold = bev._uvtr_query_proj_z8
    bev._uvtr_query_proj_z8 = _Fp16Proj(getattr(fold, "m", fold)).cuda()
    eng = bev.view_trans._zhalf_engine
    eng.keep_fp16 = True
    return {"vol_fp16": "block output read in place by the query projection and the dense engine"}
