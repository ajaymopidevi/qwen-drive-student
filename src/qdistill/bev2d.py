"""Stage 3: a 2D-BEV student that replaces the Stage-2 head's lift + voxel block +
6-layer BEV encoder, and feeds the teacher's own decoder / map / occupancy heads.

What the heads consume (perception_transformer.forward, heads.forward): ``bev_embed`` [1, 200*200, 256] (decoder
value; reshaped to [1, 256, Y, X] for the map U-Net and, as 16 ch x 16 heights, for occupancy) and the voxel volume
[1, 256, Z, Y, X], which occupancy reads only through ``uvtr_occ_proj`` (1x1x1 conv 256 -> 16). So the student
predicts two 2D maps at 200 x 200: ``bev`` (256 ch = bev_embed) and ``p16`` (256 ch = 16 ch x 16 heights = the
projected volume), and the heads run unchanged (``heads_from_bev``).

Student (``BEV2DStudent``):
  image features per camera: Stage-2 ViT-neck output (256 ch, 32 x 56) + LLM adaptor level at the same grid
  (256 ch), both frozen teacher modules -> 1x1 conv -> c_img
  lift (``Lift``): per-frame geometry from the teacher's view_trans.coord_preparing (exact per-frame lidar2img: the
  fixed rig cost the frozen head 0.6 mAP, ), depth = the frozen DepthNet softmax; feature x p(depth) summed
  into [Y, X, Z, c_img] with per-voxel segment sums (differentiable, fp32, deterministic), Z folded into channels -> [Z * c_img, Y, X]
  2D encoder: 1x1 to c_bev, ResNet basic blocks at 200 / 100 / 50, upsampled and fused back to 200 x 200
  outputs: 1x1 convs -> bev (256), p16 (256)
Layouts: BEV row = voxel y index, column = x index (teacher: kernel [B, X, Y, Z, C] -> [B, C, Z, Y, X]; bev_embed
token = y * 200 + x); p16 channel = c * 16 + z (the occupancy fuse's view(bs, -1, 16, H, W)).
third_party is not edited.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from qwen_drive_perception.layers import inverse_sigmoid


# ------------------------------------------------------------------------------------------------------------ lift
class Lift:
    """Depth-weighted lift with the cameras merged, in plain PyTorch (autograd through segment_reduce / gather)."""

    @staticmethod
    @torch.no_grad()
    def prep(voxel_coords, mask, X, Y, Z):
        """view_trans.coord_preparing output -> (point index = depth flat index, pixel flat index, voxel index)."""
        _, N_sweep, N_cam, D, H, W = mask.shape
        flat = mask.reshape(-1)
        p = torch.nonzero(flat, as_tuple=False).squeeze(1)
        c = voxel_coords.reshape(-1, 4)[p].long()
        img = p // (D * H * W)
        pix = img * (H * W) + p % (H * W)
        vox = (c[:, 2] * X + c[:, 1]) * Z + c[:, 3]  # [Y, X, Z] order
        # points sorted by voxel (stable) -> per-voxel segment sums in apply: deterministic, unlike fp32 index_add
        # atomics
        vox, order = torch.sort(vox, stable=True)
        uniq, counts = torch.unique_consecutive(vox, return_counts=True)
        cell_uniq, cell_counts = torch.unique_consecutive(vox // Z, return_counts=True)  # sorted by voxel => by cell
        return {"point": p[order], "pix": pix[order], "vox": vox, "uniq": uniq, "counts": counts,
                "z": vox % Z, "cell_uniq": cell_uniq, "cell_counts": cell_counts, "dims": (X, Y, Z, D, H, W)}

    @staticmethod
    def apply(feat, depth, prep):
        """feat [N_img, C, H, W], depth [N_img, D, H, W] -> [1, Z * C, Y, X] (fp32)."""
        X, Y, Z, D, H, W = prep["dims"]
        C = feat.shape[1]
        ft = feat.permute(0, 2, 3, 1).reshape(-1, C)              # [N_img * H * W, C]
        vals = ft[prep["pix"]].float() * depth.reshape(-1)[prep["point"]].float()[:, None]
        seg = torch.segment_reduce(vals, "sum", lengths=prep["counts"], axis=0)
        out = torch.zeros(Y * X * Z, C, device=feat.device, dtype=torch.float32).index_copy(0, prep["uniq"], seg)
        return out.view(Y, X, Z * C).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def apply_folded(g, depth, prep):
        """Lift with the following 1x1 conv folded in. g [N_img, Z * c_out, H, W] = the per-height
        slices W_z of that conv applied per pixel (channel z * c_out + o), depth [N_img, D, H, W] -> [1, c_out, Y, X]
        (fp32): sum over the points of a cell of depth * g[pixel, z(point)]. Equals apply() followed by a 1x1 conv
        with weight[o, z * C + c] = W_z[o, c]; the [Z * C, Y, X] volume is never built, so memory scales with c_out
        and not with the lifted channels."""
        X, Y, Z, D, H, W = prep["dims"]
        c_out = g.shape[1] // Z
        gt = g.permute(0, 2, 3, 1).reshape(-1, c_out)             # [N_img * H * W * Z, c_out]
        vals = gt[prep["pix"] * Z + prep["z"]].float() * depth.reshape(-1)[prep["point"]].float()[:, None]
        seg = torch.segment_reduce(vals, "sum", lengths=prep["cell_counts"], axis=0)
        out = torch.zeros(Y * X, c_out, device=g.device, dtype=torch.float32).index_copy(0, prep["cell_uniq"], seg)
        return out.view(Y, X, c_out).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def apply_2d(feat, depth, prep):
        """Z-collapsed lift: feat [N_img, C, H, W], depth [N_img, D, H, W] ->
        [1, C, Y, X] (fp32), the sum over every point of a cell whatever its height (= apply() summed over Z)."""
        X, Y, Z, D, H, W = prep["dims"]
        C = feat.shape[1]
        ft = feat.permute(0, 2, 3, 1).reshape(-1, C)
        vals = ft[prep["pix"]].float() * depth.reshape(-1)[prep["point"]].float()[:, None]
        seg = torch.segment_reduce(vals, "sum", lengths=prep["cell_counts"], axis=0)
        out = torch.zeros(Y * X, C, device=feat.device, dtype=torch.float32).index_copy(0, prep["cell_uniq"], seg)
        return out.view(Y, X, C).permute(2, 0, 1).unsqueeze(0)


# --------------------------------------------------------------------------------------------------------- student
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)), inplace=True))) + self.skip(x), inplace=True)


def conv_bn(cin, cout, k=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, 1, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BEV2DStudent(nn.Module):
    def __init__(self, c_img: int = 64, c_bev: int = 128, Z: int = 16, vit_ch: int = 256, llm_ch: int = 256,
                 blocks=(2, 2, 2), out_bev: int = 256, out_occ: int = 256, use_llm: bool = True, fold: bool = False):
        super().__init__()
        self.cfg = dict(c_img=c_img, c_bev=c_bev, Z=Z, vit_ch=vit_ch, llm_ch=llm_ch, blocks=list(blocks),
                        out_bev=out_bev, out_occ=out_occ, use_llm=use_llm, fold=fold)
        self.use_llm, self.fold, self.Z = use_llm, fold, Z
        c_in = vit_ch + (llm_ch if use_llm else 0)
        if fold:
            # folded lift: no per-pixel projection; the lift's following 1x1 conv (Z * c_in -> c_bev) runs per pixel
            # as Z height slices, then BN + ReLU in BEV (== lifting all c_in channels; c_img unused)
            self.fold_w = nn.Conv2d(c_in, Z * c_bev, 1, bias=False)
            self.bev_in = nn.Sequential(nn.BatchNorm2d(c_bev), nn.ReLU(inplace=True))
        else:
            self.img_proj = conv_bn(c_in, c_img)
            self.bev_in = conv_bn(c_img * Z, c_bev)
        c1, c2, c3 = c_bev, 2 * c_bev, 4 * c_bev
        self.s1 = nn.Sequential(*[BasicBlock(c1, c1) for _ in range(blocks[0])])
        self.s2 = nn.Sequential(BasicBlock(c1, c2, 2), *[BasicBlock(c2, c2) for _ in range(blocks[1] - 1)])
        self.s3 = nn.Sequential(BasicBlock(c2, c3, 2), *[BasicBlock(c3, c3) for _ in range(blocks[2] - 1)])
        self.lat2 = conv_bn(c2, c1)
        self.lat3 = conv_bn(c3, c1)
        self.fuse = nn.Sequential(conv_bn(3 * c1, c1, 3), BasicBlock(c1, c1))
        self.bev_out = nn.Conv2d(c1, out_bev, 1)
        self.occ_out = nn.Conv2d(c1, out_occ, 1)

    def forward(self, vit, llm, depth, prep):
        """vit / llm [N_img, 256, H, W] (frozen inputs), depth [N_img, D, H, W] -> bev [1, 256, Y, X], p16 [1, 256, Y, X]."""
        x = torch.cat([vit, llm], 1) if self.use_llm else vit
        if self.fold:
            g = self.fold_w(x)
            b = self.s1(self.bev_in(Lift.apply_folded(g, depth, prep).to(g.dtype)))
        else:
            f = self.img_proj(x)
            v = Lift.apply(f, depth, prep).to(f.dtype)
            b = self.s1(self.bev_in(v))
        b2 = self.s2(b)
        b3 = self.s3(b2)
        size = b.shape[-2:]
        u = torch.cat([b, F.interpolate(self.lat2(b2), size=size, mode="bilinear", align_corners=False),
                       F.interpolate(self.lat3(b3), size=size, mode="bilinear", align_corners=False)], 1)
        u = self.fuse(u)
        return self.bev_out(u), self.occ_out(u)


# ------------------------------------------------------------------------------------------------ encoder tail
class EncoderTail(nn.Module):
    """a3 (hybrid): teacher BEVFormer encoder layer(s) on top of the student BEV, so the BEV queries cross-attend
    into the LLM-adaptor image features as in the teacher (deformable BEV self-attention + spatial cross-attention +
    FFN). Initialised from the teacher's last ``n_layers`` layers and trained; fp32 master weights, run in bf16 (the
    ms_deform_attn kernel is bf16-only) via functional_call. Frozen: level embeddings and the learned BEV positional
    encoding. Input bev [1, 256, Y, X] (stands in for the previous layer's output), mlvl list of [N_cam, 256, h, w]
    (bev.adaptor levels) -> [1, 256, Y, X]."""

    def __init__(self, head, n_layers: int = 1, first: int | None = None):
        """``first``: index of the teacher's first layer copied.
        ``self.dtype``: bf16 (default, as deployed eagerly) or fp32 (exact math, the TensorRT engine's
        numerics; needs head_grad's differentiable dispatch, which runs fp32 through the PyTorch MSDA)."""
        super().__init__()
        self.dtype = torch.bfloat16
        from qwen_drive_perception.bev_encoder import BEVFormerEncoder
        hd = head.bev_modeling.head
        tr, enc = hd.transformer, hd.transformer.encoder
        self.n_layers = n_layers
        self.first = len(enc.layers) - n_layers if first is None else first
        self.enc = BEVFormerEncoder(num_layers=n_layers, pc_range=enc.pc_range,
                                    num_points_in_pillar=enc.num_points_in_pillar).cuda().float()
        for i, layer in enumerate(enc.layers[self.first:self.first + n_layers]):
            self.enc.layers[i].load_state_dict({k: v.float() for k, v in layer.state_dict().items()})
        self.bev_h, self.bev_w = hd.bev_h, hd.bev_w
        self.grid_length = (hd.real_h / hd.bev_h, hd.real_w / hd.bev_w)
        with torch.no_grad():
            mask = torch.zeros((1, hd.bev_h, hd.bev_w), device="cuda", dtype=torch.bfloat16)
            pos = hd.positional_encoding(mask).to(torch.bfloat16)
        self.register_buffer("bev_pos", pos.flatten(2).permute(2, 0, 1).contiguous(), persistent=False)  # [HW, 1, C]
        self.register_buffer("level_embeds", tr.level_embeds.detach().to(torch.bfloat16).clone(), persistent=False)

    @contextmanager
    def _ckpt_layers(self):
        """fp32 mode: activation checkpointing per encoder layer while grad is on (the PyTorch MSDA's fp32 sampling
        temporaries of 6 layers do not fit 8 GB)."""
        if not torch.is_grad_enabled():
            yield
            return
        saved = []
        for lay in self.enc.layers:
            f = lay.forward
            saved.append((lay, f))
            lay.forward = lambda *a, f=f, **k: checkpoint(f, *a, use_reentrant=False, **k)
        try:
            yield
        finally:
            for lay, f in saved:
                lay.forward = f

    @staticmethod
    def _metas(metas):
        """lidar2img as [N_cam, 4, 4] whatever the caller did to it (the teacher's view transform reformats it in
        place to [1, N_cam, 1, 4, 4]), rounded to bf16 as the teacher's point_sampling does (it builds the matrices
        with reference_points.new_tensor on bf16 reference points; fp32 matrices give a 5e-4 nMSE mismatch)."""
        import numpy as np
        m = dict(metas)
        l2i = m["lidar2img"]
        l2i = l2i.detach().float().cpu() if torch.is_tensor(l2i) else torch.as_tensor(np.asarray(l2i), dtype=torch.float32)
        m["lidar2img"] = l2i.reshape(-1, 4, 4).cuda().to(torch.bfloat16)
        return [m]

    def forward(self, bev2d, mlvl, metas):
        dt = self.dtype
        feats, shapes = [], []
        for lvl, f in enumerate(mlvl):
            n, c, h, w = f.shape
            x = f.to(dt)[None].flatten(3).permute(1, 0, 3, 2)  # [N, 1, hw, C]
            feats.append(x + self.level_embeds[None, None, lvl:lvl + 1, :].to(dt))
            shapes.append((h, w))
        feat = torch.cat(feats, 2).permute(0, 2, 1, 3)                   # [N, HW_img, 1, C]
        shapes = torch.as_tensor(shapes, dtype=torch.long, device=feat.device)
        start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        q = bev2d.to(dt).flatten(2).permute(2, 0, 1).contiguous()  # [HW, 1, C]
        if dt == torch.float32:  # master weights used as they are; each layer recomputed in backward (memory)
            with torch.autocast("cuda", enabled=False), self._ckpt_layers():
                out = self.enc(q, feat, feat, self.bev_h, self.bev_w, self.bev_pos.float(), shapes, start,
                               img_metas=self._metas(metas), grid_length=self.grid_length)
            return out.reshape(1, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
        params = {k: v.to(dt) for k, v in self.enc.named_parameters()}
        with torch.autocast("cuda", enabled=False):
            out = torch.func.functional_call(
                self.enc, params, (q, feat, feat, self.bev_h, self.bev_w, self.bev_pos.to(dt), shapes, start),
                {"img_metas": self._metas(metas), "grid_length": self.grid_length})
        return out.reshape(1, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------------ head path
def _ck(fn, ckpt, *args):
    return checkpoint(fn, *args, use_reentrant=False) if ckpt else fn(*args)


def heads_from_bev(bev, bev2d: torch.Tensor, p16: torch.Tensor, metas: dict, refiner: nn.Module | None,
                   occ: bool = True, ckpt: bool = False, mods: dict | None = None) -> dict:
    """The teacher's heads on the student's maps (bev2d [1, 256, Y, X] = bev_embed, p16 [1, 256, Y, X] = the
    projected voxel volume): {all_cls_scores, all_bbox_preds, seg_preds[, occ_pred]}, as pred_kd.head_from_voxel after
    the encoder. ``refiner``: a PyTorch occupancy refiner (the deployed head's occ_decoder may be a TensorRT engine).
    bf16, autocast off (as teacher.run_head). ``mods``: , trained stand-ins for "map_seg" (seg_decoder),
    "uvtr_occ_fuse", "occ_pred_head" (e.g. Fp16Run copies) instead of the head's own modules."""
    with torch.autocast("cuda", enabled=False):
        return _heads_from_bev(bev, bev2d.to(torch.bfloat16), p16.to(torch.bfloat16), metas, refiner, occ, ckpt,
                               mods or {})


class Fp16Run(nn.Module):
    """a trainable fp32-master module run as its TensorRT fp16 engine would (fp16 autocast; norms and
    softmax in fp32 per autocast), output cast back to the caller's dtype."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.m(x.float()).to(x.dtype)


def _heads_from_bev(bev, bev2d, p16, metas, refiner, occ, ckpt, mods):
    head = bev.head
    tr = head.transformer
    img_metas = [metas]
    bev_h, bev_w = head.bev_h, head.bev_w
    kw = dict(grid_length=(head.real_h / head.bev_h, head.real_w / head.bev_w), img_metas=img_metas)
    # [1, Y * X, 256], token = y * X + x; contiguous token-major like the encoder output: a channel-strided view
    # takes another GEMM path in the decoder's value projection and differs at bf16 rounding (up to 2.3 in the logits
    # after 6 layers, 2026-10-18 probe)
    bev_embed = bev2d.flatten(2).permute(0, 2, 1).contiguous()
    object_query_embeds = head.query_embedding.weight.to(bev2d.dtype)
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

        def occ_branch(feat_ego, p):
            f = tr._adapt_bev_for_occ(feat_ego, bev_h, bev_w, occ_pc_range=occ_range, occ_voxel_size=occ_vox)
            ph, pw = p.shape[-2:]                                          # p16 may be coarser than bev2d
            v = tr._adapt_volume_for_occ(p.view(1, -1, tr.occ_pillar_h, ph, pw), ph, pw,
                                         occ_pc_range=occ_range, occ_voxel_size=occ_vox)
            fuse, pred = mods.get("uvtr_occ_fuse", tr.uvtr_occ_fuse), mods.get("occ_pred_head", tr.occ_pred_head)
            f = f + fuse(torch.cat([f, v], dim=1))                         # _fuse_uvtr_occ_feat without uvtr_occ_proj
            return pred(refiner(f).permute(0, 4, 3, 2, 1))
        outs["occ_pred"] = _ck(occ_branch, ckpt, bev2d, p16)

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
    outs["seg_preds"] = mods.get("map_seg", head.seg_decoder)(head.feat_cropper(bev2d))
    return outs


# ------------------------------------------------------------------------------------------------- Stage-2 targets
class Stage2Targets:
    """Runs the deployed Stage-2 head with
    no grad on the student's taps and captures what Stage 3 needs from it:
      inputs for the 2D student: ViT-neck features, depth softmax, LLM adaptor features (all frozen), voxel geometry
      feature targets: bev_embed [1, 256, Y, X], p16 = uvtr_occ_proj(voxel block output) [1, 256, Y, X]
      output targets: all_cls_scores, all_bbox_preds, seg_preds, occ_pred (pred_kd.pred_losses format)."""

    def __init__(self, head):
        self.head = head
        bev = head.bev_modeling
        self.bev, vt, hd, tr = bev, bev.view_trans, bev.head, bev.head.transformer
        self.cap = {}
        f_vt, f_hd, f_bev = vt.forward, hd.forward, tr.get_bev_features

        def vt_forward(mlvl_feats, img_depth, img_metas):
            from qwen_drive_perception import ops  # noqa: F401
            ms = [dict(m) for m in img_metas]
            for m in ms:
                m["lidar2img"] = vt._format_lidar2img(m["lidar2img"][None, ...])
            vc, mask = vt.coord_preparing(ms)
            self.cap["vit"] = mlvl_feats[0][0].detach()
            self.cap["depth"] = img_depth[0].detach()
            self.cap["geom"] = (vc, mask)
            y = f_vt(mlvl_feats, img_depth, img_metas)
            self.cap["p16"] = tr.uvtr_occ_proj(y.to(torch.bfloat16)).flatten(1, 2).detach()  # [1, 16 * 16, Y, X]
            return y

        def hd_forward(mlvl_feats, img_metas, *a, **k):
            self.cap["llm"] = [f[0].detach() for f in mlvl_feats]
            out = f_hd(mlvl_feats, img_metas, *a, **k)
            return out

        def get_bev(*a, **k):
            e = f_bev(*a, **k)
            self.cap["bev"] = e.detach()
            return e

        vt.forward, hd.forward, tr.get_bev_features = vt_forward, hd_forward, get_bev

    @torch.no_grad()
    def __call__(self, vit_tap, llm_tap, metas):
        self.cap = {}
        bev = self.bev
        outs = bev(img_vit_feats=vit_tap.to(torch.bfloat16), img_llm_feats=llm_tap.to(torch.bfloat16), img_metas=[metas])
        c = self.cap
        h, w = bev.head.bev_h, bev.head.bev_w
        c["bev"] = c["bev"].reshape(1, h, w, -1).permute(0, 3, 1, 2).contiguous()  # get_bev_features: [1, Y * X, 256]
        c["outs"] = {k: outs[k].detach() for k in ("all_cls_scores", "all_bbox_preds", "seg_preds", "occ_pred")}
        return c


# ------------------------------------------------------------------------------------------------ deployed Stage 3
