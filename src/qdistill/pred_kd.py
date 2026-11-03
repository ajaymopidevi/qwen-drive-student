"""Prediction-level KD through the frozen head.

The student's block output and the teacher's go through the SAME frozen head, so their predictions line up query for
query (900 detection queries, BEV map cells, occupancy voxels): no matching is needed.

``head_from_voxel`` is upstream's forward after view_trans (modeling_perception.forward -> heads.forward ->
perception_transformer.forward) with two switches: ``occ`` (the occupancy refiner, the largest backward) and
``ckpt`` (activation checkpointing of the BEV encoder, decoder and refiner: recomputed in backward). With occ=True
and ckpt=False it equals upstream exactly. Needs head_grad.enable_head_grad()
for the backward through deformable attention. third_party is not edited.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from qwen_drive_perception.layers import inverse_sigmoid


@torch.no_grad()
def llm_mlvl(bev, llm: torch.Tensor) -> list[torch.Tensor]:
    """Adaptor features of the LLM tap ``llm`` [N_cam, H, W, C] -> 4 x [1, N, 256, h, w] (no grad: frozen input)."""
    n = llm.shape[0]
    return [f.view(1, n, *f.shape[1:]).to(torch.bfloat16) for f in bev.adaptor(llm.permute(0, 3, 1, 2))]


def _ck(fn, ckpt, *args, **kw):
    return checkpoint(fn, *args, use_reentrant=False, **kw) if ckpt else fn(*args, **kw)


def head_from_voxel(bev, mlvl: list[torch.Tensor], y: torch.Tensor, metas: dict, occ: bool = True, ckpt: bool = False) -> dict:
    """Frozen head on block output ``y`` [1, 256, Z, Y, X]: {all_cls_scores, all_bbox_preds, seg_preds[, occ_pred]}.
    Runs like teacher.run_head: bf16 weights and inputs, autocast off (under autocast the spatial cross-attention mixes
    fp32 and bf16 and its scatter_add_ fails)."""
    with torch.autocast("cuda", enabled=False):
        return _head_from_voxel(bev, [f.to(torch.bfloat16) for f in mlvl], y.to(torch.bfloat16), metas, occ, ckpt)


def _head_from_voxel(bev, mlvl, y, metas, occ, ckpt):
    head = bev.head
    tr = head.transformer
    img_metas = [metas]
    dtype = mlvl[0].dtype
    bev_queries = head.bev_embedding.weight.to(dtype) + bev._uvtr_voxel_to_bev_tokens(y).to(dtype)
    object_query_embeds = head.query_embedding.weight.to(dtype)
    bev_mask = torch.zeros((1, head.bev_h, head.bev_w), device=bev_queries.device).to(dtype)
    bev_pos = head.positional_encoding(bev_mask).to(dtype)
    kw = dict(grid_length=(head.real_h / head.bev_h, head.real_w / head.bev_w), img_metas=img_metas)  # the encoder sets prev_bev itself
    bev_h, bev_w = head.bev_h, head.bev_w

    bev_embed = _ck(lambda q: tr.get_bev_features(mlvl, q, bev_h, bev_w, bev_pos=bev_pos, **kw), ckpt, bev_queries)
    bev_feat_ego = bev_embed.reshape(1, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()

    query_pos, query = torch.split(object_query_embeds, tr.embed_dims, dim=1)
    query_pos, query = query_pos.unsqueeze(0), query.unsqueeze(0)
    reference_points = tr.reference_points(query_pos).sigmoid()
    spatial = query.new_tensor([[bev_h, bev_w]], dtype=torch.long)

    def decode(value):
        return tr.decoder(query=query.permute(1, 0, 2), key=None, value=value, query_pos=query_pos.permute(1, 0, 2),
                          reference_points=reference_points, reg_branches=head.reg_branches, cls_branches=None,
                          spatial_shapes=spatial, level_start_index=spatial.new_zeros((1,)), **kw)
    hs, inter_references = _ck(decode, ckpt, bev_embed.permute(1, 0, 2))

    outs = {}
    if occ:
        occ_range, occ_vox = tr._get_occ_target_spec(img_metas)

        def occ_branch(feat_ego, vol):
            f = tr._adapt_bev_for_occ(feat_ego, bev_h, bev_w, occ_pc_range=occ_range, occ_voxel_size=occ_vox)
            v = tr._adapt_volume_for_occ(vol, bev_h, bev_w, occ_pc_range=occ_range, occ_voxel_size=occ_vox)
            f = tr._fuse_uvtr_occ_feat(f, v)
            return tr.occ_pred_head(tr.occ_decoder(f).permute(0, 4, 3, 2, 1))
        outs["occ_pred"] = _ck(occ_branch, ckpt, bev_feat_ego, y)

    hs = hs.permute(0, 2, 1, 3)
    classes, coords = [], []
    pc = head.pc_range
    for lvl in range(hs.shape[0]):  # heads.forward's per-layer decoding, verbatim
        reference = inverse_sigmoid(reference_points if lvl == 0 else inter_references[lvl - 1])
        cls = head.cls_branches[lvl](hs[lvl])
        tmp = head.reg_branches[lvl](hs[lvl])
        tmp[..., 0:2] += reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] += reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        tmp[..., 0:1] = tmp[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        tmp[..., 1:2] = tmp[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        tmp[..., 4:5] = tmp[..., 4:5] * (pc[5] - pc[2]) + pc[2]
        classes.append(cls)
        coords.append(tmp)
    outs["all_cls_scores"] = torch.stack(classes)
    outs["all_bbox_preds"] = torch.stack(coords)
    outs["seg_preds"] = head.seg_decoder(head.feat_cropper(bev_feat_ego))
    return outs


def pred_losses(s: dict, t: dict, class_weight: torch.Tensor | None = None, query_norm: float | None = None,
                box_class_weight: torch.Tensor | None = None, box_dim_weight: torch.Tensor | None = None,
                occ_fg_weight: float = 1.0) -> dict:
    """Student outputs ``s`` vs teacher outputs ``t`` (same frame, same head), last decoder layer:
    det_cls  soft-target BCE minus the teacher's own entropy (0 at a match), per query summed over classes, averaged with
             query weights w = teacher max prob + 0.01 (the confident queries are the detections); w is normalised to
             sum 1 per frame, or divided by the fixed ``query_norm`` (the mean per-frame sum, 68 on
             s1_C4k, so a frame with many detections, e.g. a row of barriers, keeps proportionally more weight);
             ``class_weight`` [7] scales each class's term
    det_box  L1 on the decoded box parameters, each dimension divided by its teacher std over queries, same w;
             ``box_class_weight`` [7] scales each query by its teacher argmax class (renormalised to the same sum of w),
             ``box_dim_weight`` [10] weights the dimensions (cx cy w l cz h sin cos vx vy; normalised to mean 1)
    map      KL(teacher || student) of the per-cell 6-class softmax, mean over cells
    occ      the same over occupancy voxels (when present); ``occ_fg_weight`` != 1 weights each voxel by
             1 + (occ_fg_weight - 1) x (1 - teacher p(empty)) and takes the weighted mean, so occupied voxels are not
             drowned by empty ones (empty = last class)."""
    out = {}
    ls, lt = s["all_cls_scores"][-1, 0].float(), t["all_cls_scores"][-1, 0].float()  # [Q, 7]
    pt = lt.sigmoid()
    bce = F.binary_cross_entropy_with_logits(ls, pt, reduction="none") - F.binary_cross_entropy_with_logits(lt, pt, reduction="none")
    if class_weight is not None:
        bce = bce * class_weight.to(bce)
    w = pt.max(-1).values + 0.01
    w = (w / (w.sum() if query_norm is None else query_norm)).detach()
    out["det_cls"] = (bce.sum(-1) * w).sum()
    bs, bt = s["all_bbox_preds"][-1, 0].float(), t["all_bbox_preds"][-1, 0].float()  # [Q, 10]
    e = (bs - bt).abs() / bt.std(0).clamp_min(1e-3)
    if box_dim_weight is not None:
        dw = box_dim_weight.to(e)
        e = e * (dw / dw.mean())
    wb = w
    if box_class_weight is not None:
        wb = w * box_class_weight.to(w)[lt.argmax(-1)]
        wb = (wb * (w.sum() / wb.sum())).detach()
    out["det_box"] = (e.mean(-1) * wb).sum()
    for k, key, dim in (("map", "seg_preds", 1), ("occ", "occ_pred", -1)):
        if key in s:
            lt_ = t[key].float().log_softmax(dim)
            kl = F.kl_div(s[key].float().log_softmax(dim), lt_, log_target=True, reduction="none").sum(dim)
            if k == "occ" and occ_fg_weight != 1.0:
                wv = (1 + (occ_fg_weight - 1) * (1 - lt_[..., -1].exp())).detach()
                out[k] = (kl * wv).sum() / wv.sum()
            else:
                out[k] = kl.mean()
    return out
