"""the detection queries read the cameras themselves (Sparse4D-style keypoint sampling).

    dec = SamplingDecoder(head)                    # fp32 copy of the teacher's 6 decoder layers + one sampler per layer
    install(head, dec)                             # replaces head.transformer.decoder; levels from head.forward's input
    install(head, "runs/sampler/final.pt")          # or head_chain "det_sample:<path>"

With the dense path's image reads gone, the BEV embedding carries no camera evidence the encoder used to read
per cell. Each decoder layer gets that evidence back per query, after its BEV deformable cross-attention and before
its norm + FFN (the post-norm order self_attn -> norm -> cross_attn -> [+ sampler] -> norm -> ffn -> norm):
  keypoints  7 fixed points of the query's current box (centre, the 6 face centres) + K learned offsets
             (tanh(Linear(query)) in box units, |offset| <= 1 box size per axis), in the box frame (x along the heading,
             sizes exp(w) x exp(l) x exp(h): code [cx cy w l cz h sin cos vx vy], w along the heading as the packed GT,
             nuscenes_pack.py). The box is the previous layer's prediction (detached, as the reference points); layer 0
             uses the reference point with a 2 x 2 x 2 m box at yaw 0
  sampling   ego -> image with lidar2img @ inv(lidar2ego) (fp32), bilinear grid_sample of each adaptor level per
             camera; points behind the camera or outside the image contribute 0
  weights    softmax over (keypoint, camera, level) per channel group (8 groups of 32), predicted from the query
  output     Linear(256 -> 256), zero-initialised: an untrained sampler returns 0 and the decoder equals the teacher's
Per frame ~900 x (7 + K) keypoints x 6 cameras x 4 levels bilinear reads (K 6: 280k) vs the encoder's per-cell reads
(40,000 cells x 4 pillar points x 8 x 4 levels, on the cameras each cell projects to: 5.1M on 1 camera).
fp32 master weights, run in bf16 via functional_call (the ms_deform_attn kernel is bf16-only), as bev2d.EncoderTail.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qwen_drive_perception.layers import inverse_sigmoid

# what the samplers read. "llm" the 4 LLM-adaptor levels; "vit" / "vit3" the head's ViT-neck output
# [N, 256, 32, 56] (the lift's input, computed anyway), alone or with its 2 x 2 / 4 x 4 average pools (16 x 28, 8 x 14):
# the adaptor's levels 1-3 at the same strides, so an "llm" sampler initialises a "vit*" one (adapt_state).
LEVELS = {"vit": (1,), "vit3": (1, 2, 3)}


def vit_levels(v, mode: str) -> list:
    """ViT-neck output [N, C, 32, 56] -> the sampler's levels for ``mode`` ("vit" / "vit3")."""
    return [v] if mode == "vit" else [v, F.avg_pool2d(v, 2), F.avg_pool2d(v, 4)]


FIXED = ((0, 0, 0), (0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0), (0, 0, 0.5), (0, 0, -0.5))
DEFAULT_SIZE = (2.0, 2.0, 2.0)


class QueryImageSampler(nn.Module):
    def __init__(self, c: int = 256, n_learn: int = 6, n_cam: int = 6, n_levels: int = 4, groups: int = 8):
        super().__init__()
        self.c, self.n_learn, self.n_cam, self.n_levels, self.groups = c, n_learn, n_cam, n_levels, groups
        self.k = len(FIXED) + n_learn
        self.register_buffer("fixed", torch.tensor(FIXED, dtype=torch.float32), persistent=False)
        self.offsets = nn.Linear(c, n_learn * 3)
        self.weights = nn.Linear(c, groups * self.k * n_cam * n_levels)
        self.out = nn.Linear(c, c)
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():  # learned points start spread over the box (tanh(1.5 x +-1) ~ +-0.9 box size)
            self.offsets.bias.copy_(torch.empty(n_learn * 3).uniform_(-1.5, 1.5))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def keypoints(self, q, box):
        """q [B, Q, C]; box (centre [B, Q, 3] m, size [B, Q, 3] m, sincos [B, Q, 2]) -> ego points [B, Q, K, 3]."""
        centre, size, sc = box
        b, n, _ = q.shape
        learn = torch.tanh(self.offsets(q).float()).view(b, n, self.n_learn, 3)
        unit = torch.cat([self.fixed.expand(b, n, -1, -1), learn], 2)                    # box units [B, Q, K, 3]
        local = unit * size[:, :, None, :]
        s, c = sc[..., 0:1], sc[..., 1:2]                                                 # [B, Q, 1]
        x = local[..., 0] * c - local[..., 1] * s
        y = local[..., 0] * s + local[..., 1] * c
        return centre[:, :, None, :] + torch.stack([x, y, local[..., 2]], -1)

    def _project(self, q, box, ego2img, img_hw):
        """-> uv [N, Q K, 2] in [0, 1] image coordinates, valid [N, Q K], softmax weights [Q, G, K, N, L] (fp32)."""
        n = q.shape[1]
        pts = self.keypoints(q, box)[0].reshape(-1, 3)                                   # [Q K, 3]
        hom = torch.cat([pts, torch.ones_like(pts[:, :1])], -1)
        cam = torch.einsum("nij,pj->npi", ego2img, hom)                                   # [N, Q K, 4]
        z = cam[..., 2:3]
        uv = cam[..., :2] / z.clamp_min(1e-5)
        uv = torch.stack([uv[..., 0] / img_hw[1], uv[..., 1] / img_hw[0]], -1)
        valid = (z[..., 0] > 1e-5) & (uv > 0).all(-1) & (uv < 1).all(-1)
        w = self.weights(q[0]).view(n, self.groups, self.k * self.n_cam * self.n_levels).float().softmax(-1)
        return uv.clamp(-1, 2), valid, w.view(n, self.groups, self.k, self.n_cam, self.n_levels)

    def forward(self, q, box, value, ego2img, img_hw):
        """q [B=1, Q, C] (query + query_pos), value = prepare_value(levels), ego2img [N_cam, 4, 4] fp32, img_hw (H, W)
        -> [B, Q, C] (as q). The upstream ms_deform_attn kernel (bf16, differentiable with head_grad) with each
        (camera, level) map as one of N x L levels: sampling and the weighted sum fused, no [N, L, C, Q K] tensor."""
        from qwen_drive_perception import attention
        n, g = q.shape[1], self.groups
        uv, valid, w = self._project(q, box, ego2img, img_hw)
        loc = uv.view(self.n_cam, n, self.k, 2).permute(1, 0, 2, 3)                        # [Q, N, K, 2]
        loc = loc[:, :, None].expand(n, self.n_cam, self.n_levels, self.k, 2).reshape(n, 1, -1, self.k, 2)
        loc = loc.expand(n, g, self.n_cam * self.n_levels, self.k, 2)[None]                # [1, Q, G, N L, K, 2]
        m = valid.view(self.n_cam, n, self.k).permute(1, 0, 2)[:, None, :, None, :]        # [Q, 1, N, 1, K]
        w = (w.permute(0, 1, 3, 4, 2) * m).reshape(1, n, g, self.n_cam * self.n_levels, self.k)
        v, shapes, start = value
        agg = attention.multi_scale_deformable_attn_cuda(v, shapes, start, loc.to(v.dtype).contiguous(),
                                                         w.to(v.dtype).contiguous(), 64)  # [1, Q, C]
        return self.out(agg.to(q.dtype))

    def reference(self, q, box, levels, ego2img, img_hw):
        """The same output with grid_sample per level (levels 4 x [N_cam, C, h, w]): test reference, ~6x the memory."""
        n = q.shape[1]
        uv, valid, w = self._project(q, box, ego2img, img_hw)
        grid = (uv * 2 - 1).to(q.dtype)[:, :, None, :]                                    # [N, Q K, 1, 2]
        s = torch.stack([F.grid_sample(f.to(q.dtype), grid, mode="bilinear", padding_mode="zeros",
                                       align_corners=False)[..., 0] for f in levels], 1)
        s = (s * valid[:, None, None, :].to(q.dtype)).view(self.n_cam, self.n_levels, self.groups, -1, n, self.k)
        agg = torch.einsum("qgknl,nlgcqk->qgc", w.to(q.dtype), s).reshape(1, n, self.c)
        return self.out(agg)


def prepare_value(levels, groups: int = 8):
    """levels 4 x [N_cam, C, h, w] -> (value [1, sum N h w, G, C / G], spatial shapes [N L, 2], level start [N L]),
    level index = camera x L + level (the order QueryImageSampler.forward builds its locations in)."""
    n_cam, c = levels[0].shape[:2]
    flat, shapes = [], []
    for i in range(n_cam):
        for f in levels:
            flat.append(f[i].flatten(1).T)                                                  # [h w, C]
            shapes.append(f.shape[2:])
    v = torch.cat(flat, 0).contiguous().view(1, -1, groups, c // groups)
    shapes = torch.as_tensor(shapes, dtype=torch.long, device=v.device)
    start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    return v, shapes, start


class _Core(nn.Module):
    """The decoder loop of DetectionTransformerDecoder with the sampler between each layer's cross-attention and its
    second norm; BaseTransformerLayer's calls (operation order, arguments) are reproduced exactly."""

    def __init__(self, layers, samplers, pc_range):
        super().__init__()
        self.layers, self.samplers, self.pc_range = layers, samplers, list(pc_range)

    def _const(self, vals, like):
        """``like.new_tensor(vals)``, cached per (values, dtype, device): no host copy per call (CUDA-graph capture,
        qdistill.graph_decoder); the same values and dtype as new_tensor."""
        cache = self.__dict__.setdefault("_consts", {})
        key = (tuple(vals), like.dtype, like.device)
        if key not in cache:
            cache[key] = like.new_tensor(vals)
        return cache[key]

    def _box(self, ref, tmp):
        pc = self.pc_range
        lo = self._const(pc[:3], ref)
        centre = ref.float() * (self._const(pc[3:], ref) - lo) + lo
        if tmp is None:
            size = self._const(DEFAULT_SIZE, centre).expand_as(centre)
            sc = self._const((0.0, 1.0), centre).expand(*centre.shape[:-1], 2)
        else:
            t = tmp.detach().float()
            size = torch.stack([t[..., 2], t[..., 3], t[..., 5]], -1).clamp(-3, 4).exp()
            sc = F.normalize(t[..., 6:8], dim=-1)
        return centre, size, sc

    def forward(self, query, value, query_pos, reference_points, reg_branches, spatial_shapes, level_start_index,
                img_value, ego2img, img_hw, kw):
        output, tmp = query, None
        intermediate, refs = [], []
        for lid, layer in enumerate(self.layers):
            ref_in = reference_points[..., :2].unsqueeze(2)
            k = dict(kw, reference_points=ref_in, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
            q = layer.attentions[0](output, output, output, None, query_pos=query_pos, key_pos=query_pos, **k)
            q = layer.norms[0](q)
            q = layer.attentions[1](q, None, value, None, query_pos=query_pos, key_pos=None, **k)
            if self.samplers[lid] is not None:                                            # [Q, B, C] layout
                box = self._box(reference_points, tmp)
                q = q + self.samplers[lid]((q + query_pos).permute(1, 0, 2), box, img_value, ego2img, img_hw).permute(1, 0, 2)
            q = layer.norms[1](q)
            q = layer.ffns[0](q, None)
            output = layer.norms[2](q)
            o = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](o)
                new = torch.zeros_like(reference_points)
                new[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = new.sigmoid().detach()
            intermediate.append(output)
            refs.append(reference_points)
        return torch.stack(intermediate), torch.stack(refs)


def ego2img_of(metas) -> tuple[torch.Tensor, tuple[int, int]]:
    """lidar2img @ inv(lidar2ego) [N_cam, 4, 4] fp32 on cuda (lidar2img in any of the shapes the head leaves it in)."""
    l2i, l2e = metas["lidar2img"], metas["lidar2ego"]
    l2i = l2i.detach().float().cpu() if torch.is_tensor(l2i) else torch.as_tensor(np.asarray(l2i), dtype=torch.float32)
    l2e = l2e.detach().float().cpu() if torch.is_tensor(l2e) else torch.as_tensor(np.asarray(l2e), dtype=torch.float32)
    l2i, l2e = l2i.reshape(-1, 4, 4).double(), l2e.reshape(-1, 4, 4)[0].double()
    h, w = metas["img_shape"][0]
    return (l2i @ torch.linalg.inv(l2e)).float().cuda(), (int(h), int(w))


class SamplingDecoder(nn.Module):
    """Stands in for head.transformer.decoder (same call and outputs). ``levels`` (4 x [N_cam, C, h, w], the adaptor
    output of this frame) must be set before each call: by the caller (training) or by install()'s head.forward hook."""

    def __init__(self, head, n_learn: int = 6, sample_layers=(0, 1, 2, 3, 4, 5), groups: int = 8, levels: str = "llm"):
        super().__init__()
        assert levels == "llm" or levels in LEVELS, levels
        hd = head.bev_modeling.head
        dec = hd.transformer.decoder
        layers = copy.deepcopy(dec.layers).float()
        n_cam = 6
        n_lv = hd.transformer.num_feature_levels if levels == "llm" else len(LEVELS[levels])
        samplers = nn.ModuleList(QueryImageSampler(hd.embed_dims, n_learn, n_cam, n_lv, groups)
                                 if i in sample_layers else None for i in range(len(layers)))
        self.core = _Core(layers, samplers, hd.pc_range).cuda()
        self.cfg = dict(n_learn=n_learn, sample_layers=tuple(sample_layers), groups=groups)
        if levels != "llm":  #
            self.cfg["levels"] = levels
        self.levels = None

    def forward(self, query, value, query_pos, reference_points, reg_branches=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        assert self.levels is not None, "SamplingDecoder: set .levels (adaptor output of this frame) first"
        kw = dict(kwargs)
        kw.pop("key", None)
        ego2img, hw = ego2img_of(kw["img_metas"][0])
        dtype = value.dtype
        params = {k: v.to(dtype) for k, v in self.core.named_parameters()}
        buffers = {k: v for k, v in self.core.named_buffers()}
        with torch.autocast("cuda", enabled=False):
            return torch.func.functional_call(
                self.core, {**params, **buffers},
                (query, value, query_pos, reference_points, reg_branches, spatial_shapes, level_start_index,
                 prepare_value([f.to(dtype) for f in self.levels], self.cfg["groups"]), ego2img, hw, kw))

    def trainable(self, decoder_layers: bool = True):
        """(decoder-layer params, sampler params)."""
        lay = list(self.core.layers.parameters()) if decoder_layers else []
        return lay, [p for s in self.core.samplers if s is not None for p in s.parameters()]

    def state(self) -> dict:
        return {"core": {k: v.detach().cpu() for k, v in self.core.state_dict().items()}, "cfg": self.cfg}


def install(head, dec_or_path) -> dict:
    """Swap head.transformer.decoder for a SamplingDecoder (or the one saved at a path), and hook head.forward so each
    frame's adaptor levels reach it (in place, permanent)."""
    hd = head.bev_modeling.head
    if isinstance(dec_or_path, (str, Path)):
        ck = torch.load(dec_or_path, map_location="cpu", weights_only=False)
        dec = SamplingDecoder(head, **ck["cfg"])
        dec.core.load_state_dict(ck["core"])
        dec.eval().requires_grad_(False)
        info = {"det_sample": str(dec_or_path), **ck["cfg"]}
    else:
        dec, info = dec_or_path, dict(dec_or_path.cfg)
    hd.transformer.decoder = dec
    mode = dec.cfg.get("levels", "llm")
    if mode == "llm":
        f = hd.forward

        def forward(mlvl_feats, img_metas, *a, **k):
            dec.levels = [m[0] for m in mlvl_feats]                                      # [B=1, N, C, h, w] -> [N, C, h, w]
            return f(mlvl_feats, img_metas, *a, **k)
        hd.forward = forward
    else:  # the ViT-neck output of this frame (give det_sample after mod_trt: the engine is what gets wrapped)
        bev = head.bev_modeling
        bev.vit_neck = _VitNeckTap(bev.vit_neck, dec, mode)
    info["params_M"] = round(sum(p.numel() for s in dec.core.samplers if s is not None for p in s.parameters()) / 1e6, 2)
    return info


class _VitNeckTap(nn.Module):
    """wraps bev_modeling.vit_neck; hands each frame's output (levels for the sampler mode) to the decoder."""

    def __init__(self, inner, dec, mode: str):
        super().__init__()
        self.inner, self.mode = inner, mode
        self.dec = [dec]  # a list: not a registered submodule

    def forward(self, x):
        out = self.inner(x)
        self.dec[0].levels = vit_levels(out[0], self.mode)                               # [N, C, 32, 56] (B = 1)
        return out


def adapt_state(core_sd: dict, cfg_old: dict, dec: SamplingDecoder) -> dict:
    """an "llm" sampler checkpoint's core state -> ``dec``'s ("vit" / "vit3"): every weight as it is except the
    softmax-weight Linears, whose (keypoint, camera, level) rows are kept for the adaptor levels at the ViT strides."""
    mode = dec.cfg.get("levels", "llm")
    assert cfg_old.get("levels", "llm") == "llm" and mode in LEVELS, (cfg_old, dec.cfg)
    assert {k: v for k, v in cfg_old.items() if k != "levels"} == {k: v for k, v in dec.cfg.items() if k != "levels"}
    idx = list(LEVELS[mode])
    out = dict(core_sd)
    for name, smp in ((n, m) for n, m in dec.core.named_modules() if isinstance(m, QueryImageSampler)):
        for part in ("weight", "bias"):
            key = f"{name}.weights.{part}"
            w = core_sd[key]
            w = w.view(smp.groups, smp.k, smp.n_cam, -1, *w.shape[1:])[:, :, :, idx]
            out[key] = w.reshape(-1, *w.shape[4:]).contiguous()
    return out
