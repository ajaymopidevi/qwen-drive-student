"""warped BEV grid (varied cell size): 0.512 m cells within +-b of the ego, 1.024 m beyond, the same in X and Y on
both sides of the ego; Z (16 levels) unchanged. b = 51.2 - 1.024 k: 25.6 m -> 150 x 150, 20.48 m -> 140 x 140,
51.2 m -> 200 x 200 = (identity, the self-test of every patched site).

    python -m qdistill.warp_grid self-test                      # CPU: warp math, map / occupancy grids at identity
    python -m qdistill.warp_grid block-engine --b 25.6 --out outputs/e108/blk_w150_fp16.engine   # r64 block at G x G
    python -m qdistill.warp_grid check --chain "<chain>" --voxel-engine <G engine> --b 51.2   # identity on GPU
    head_chain "...,warp_grid:<b>:<G voxel engine>[:<final.pt>],enc_trt:<G encoder engine>,..."  (before enc_trt)
    sw = Switch(head, <G voxel engine>, b)                      # training: train_enc_recover --warp-grid, refit

Warped on both sides of the pipeline:
  write  pooling: the fine voxel index i in [0, 200) (fast_geom / coord_preparing, unchanged) is remapped to the warped
         index j(i) before the merged-pool kernel, which writes the G x G x 16 volume directly. A warped cell is the exact
         sum of its fine cells; the block reads it times n = 1 / (fine cells in it) (1, 1/2, 1/4: the per-0.512 m-cell
         mean the block was trained on). BEV queries and row / column positional embeddings
         start as the mean of their fine cells; the BEV training target is warped the same way (Switch.to_half).
  read   detection decoder: a forward pre-hook on its BEV deformable cross-attention maps the reference points
         (normalised over +-51.2 m) to the warped grid (fires only when the value is a G x G BEV, so the Stage-2 target
         runs at 200 x 200 in training are untouched); box regression / camera sampler stay metric. Map cropper: its
         cached sampling grid is warped. Occupancy: PerceptionTransformer._normalized_occ_grid warps X / Y when called
         for a G x G source (covers both the BEV and the voxel-volume read).
Unchanged: the encoder (no image reads, cell-centre reference points, offsets in cells), uvtr_query_proj,
everything on the map / occupancy / image grids.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
FINE, CELL, HALF = 200, 0.512, 51.2
BLOCK_ONNX = ROOT / "weights/onnx/voxel_r64.onnx"                       # deployed r64 block, dynamic X / Y / Z


class Warp:
    """Per-axis warp in fine-cell units: t = (x + 51.2) / 0.512 in [0, 200] -> u in [0, G] (cell k = [k, k + 1))."""

    def __init__(self, b: float = 25.6):
        nf = round(b / CELL)
        assert abs(nf * CELL - b) < 1e-6 and 0 < nf <= FINE // 2 and (FINE // 2 - nf) % 2 == 0, \
            f"b = {b}: must be 51.2 - 1.024 k"
        self.b, self.nf = b, nf
        self.t0, self.t1 = FINE // 2 - nf, FINE // 2 + nf                    # fine zone [t0, t1) in fine cells
        self.kc = self.t0 // 2                                                 # coarse cells per side
        self.G = 2 * self.kc + 2 * nf

    def index(self, device=None) -> torch.Tensor:
        """Fine index i [200] -> warped index j (long)."""
        i = torch.arange(FINE, device=device)
        return torch.where(i < self.t0, i // 2, torch.where(i < self.t1, self.kc + i - self.t0,
                                                            self.kc + 2 * self.nf + (i - self.t1) // 2))

    def u(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous fine coordinate -> continuous warped coordinate (piecewise linear, monotone, same knots)."""
        if not self.kc:                                                        # identity: no ring, slope 1 beyond
            return t - self.t0                                                 # the range too (sampling points < 0)
        return torch.where(t < self.t0, t / 2, torch.where(t < self.t1, self.kc + (t - self.t0),
                                                           self.kc + 2 * self.nf + (t - self.t1) / 2))

    def u_inv(self, u: torch.Tensor) -> torch.Tensor:
        """Inverse of u (warped -> fine continuous coordinate)."""
        if not self.kc:
            return u + self.t0
        return torch.where(u < self.kc, u * 2, torch.where(u < self.kc + 2 * self.nf, self.t0 + (u - self.kc),
                                                           self.t1 + (u - self.kc - 2 * self.nf) * 2))

    def locations(self, ref: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        """deformable sampling locations on the warped grid with the metric geometry. ``ref``: reference
        points normalised over +-51.2 m (unwarped); ``off``: sampling offsets in fine (0.512 m) cells
        decoder learned them (offset_normalizer 200). Point = u(ref * 200 + off) / G: exact on both sides of a knot."""
        return self.u(ref.float() * FINE + off.float()) / self.G

    def counts(self, device=None) -> torch.Tensor:
        return torch.bincount(self.index(device), minlength=self.G)

    def avg(self, device=None) -> torch.Tensor:
        """[G, 200] averaging matrix: row j = mean over the fine cells mapped to j."""
        j = self.index(device)
        a = torch.zeros(self.G, FINE, device=device)
        a[j, torch.arange(FINE, device=device)] = 1.0
        return a / a.sum(1, keepdim=True)

    def norm_ac(self, x: torch.Tensor) -> torch.Tensor:
        """Metric x (m) -> grid_sample coordinate with align_corners=True on the warped grid (-1 / 1 = the outer cell
        centres)."""
        return 2.0 * (self.u((x + HALF) / CELL) - 0.5) / (self.G - 1) - 1.0

    def cell_norm(self, device=None) -> torch.Tensor:
        """[G, G] 1 / (fine cells in each warped cell)."""
        c = self.counts(device).float()
        return 1.0 / (c[:, None] * c[None, :])


# ------------------------------------------------------------------------------------------------------- head state
def _state(head) -> dict:
    hd = head.bev_modeling.head
    pe = hd.positional_encoding
    return {"bev_embedding": hd.bev_embedding, "row_embed": pe.row_embed, "col_embed": pe.col_embed,
            "bev_h": hd.bev_h, "bev_w": hd.bev_w, "_map_grid": hd.feat_cropper._map_grid}


def _apply(head, st: dict):
    hd = head.bev_modeling.head
    pe = hd.positional_encoding
    hd.bev_embedding, pe.row_embed, pe.col_embed = st["bev_embedding"], st["row_embed"], st["col_embed"]
    hd.bev_h, hd.bev_w = st["bev_h"], st["bev_w"]
    hd.feat_cropper._map_grid = st["_map_grid"]


def _rows(e: nn.Embedding, a: torch.Tensor) -> nn.Embedding:
    out = nn.Embedding(a.shape[0], e.weight.shape[1]).to(device=e.weight.device, dtype=e.weight.dtype)
    out.weight.data.copy_((a.to(e.weight.device) @ e.weight.data.float()).to(e.weight.dtype))
    return out


def warp_tokens(w: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """[200 * 200, C] (token = y * 200 + x) -> [G * G, C], mean of each warped cell's fine cells."""
    c = w.shape[1]
    g = w.float().reshape(FINE, FINE, c)
    a = a.to(w.device)
    return torch.einsum("gy,yxc,hx->ghc", a, g, a).reshape(-1, c).to(w.dtype)


def warp_bev(bev: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """[B, C, 200, 200] -> [B, C, G, G], cell means (training targets)."""
    a = a.to(bev.device)
    return (a @ bev.float() @ a.T).to(bev.dtype)


def map_grid(fc, w: Warp, device) -> torch.Tensor:
    """The map cropper's sampling grid (heads.BevFeatureSlicer._grid) on the warped det grid."""
    sx0, sx1, dx, sy0, sy1, dy, _, _ = fc._map_params
    xs = torch.arange(sx0, sx1, dx, device=device)
    ys = torch.arange(sy0, sy1, dy, device=device)
    gy, gx = torch.meshgrid(w.norm_ac(ys), w.norm_ac(xs), indexing="ij")
    return torch.stack((gx, gy), dim=2)


def warp_state(head, w: Warp, queries: torch.Tensor | None = None) -> dict:
    hd = head.bev_modeling.head
    assert (hd.bev_h, hd.bev_w) == (FINE, FINE), (hd.bev_h, hd.bev_w)
    dev = hd.bev_embedding.weight.device
    a = w.avg(dev)
    emb = hd.bev_embedding
    new = nn.Embedding(w.G * w.G, emb.weight.shape[1]).to(device=dev, dtype=emb.weight.dtype)
    new.weight.data.copy_((queries if queries is not None else warp_tokens(emb.weight.data, a)).to(new.weight.dtype))
    new.requires_grad_(False)
    pe = hd.positional_encoding
    return {"bev_embedding": new, "row_embed": _rows(pe.row_embed, a), "col_embed": _rows(pe.col_embed, a),
            "bev_h": w.G, "bev_w": w.G, "_map_grid": map_grid(hd.feat_cropper, w, "cuda")}


# ------------------------------------------------------------------------------- read-side patches (shape-gated)
def _patch_occ(head, w: Warp):
    tr = head.bev_modeling.head.transformer
    if getattr(tr, "_warp_occ", None) is not None:
        return
    f = tr._normalized_occ_grid

    def occ_grid(bev_h, bev_w, device, dtype, occ_pc_range=None, occ_voxel_size=None):
        if (bev_h, bev_w) != (w.G, w.G):
            return f(bev_h, bev_w, device, dtype, occ_pc_range=occ_pc_range, occ_voxel_size=occ_voxel_size)
        # base grid in the caller's dtype computes it (fp16 arithmetic there is up to 0.15 cell off fp32; an
        # fp32 base flipped 0.3-0.5% of occ labels at identity); the warp itself runs in fp32
        g, th, tw = f(FINE, FINE, device, dtype, occ_pc_range=occ_pc_range, occ_voxel_size=occ_voxel_size)
        g = g.float()
        t = (g[..., :2] + 1.0) * (FINE - 1) / 2.0 + 0.5                    # fine continuous coordinate
        n = 2.0 * (w.u(t) - 0.5) / (w.G - 1) - 1.0
        return torch.cat([n, g[..., 2:]], -1).to(dtype), th, tw
    tr._normalized_occ_grid = occ_grid
    tr._warp_occ = w


def _patch_decoder(head, w: Warp) -> int:
    """Pre-hook on every BEV cross-attention of the installed decoder (copies made later, e.g. SamplingDecoder, carry
    the hook with them)."""
    from qwen_drive_perception.attention import CustomMSDeformableAttention

    n_tok = w.G * w.G

    def hook(mod, args, kwargs):
        if getattr(mod, "_warp_offsets", None) is not None:                     # the forward warps instead
            return None
        v = args[2] if len(args) > 2 else kwargs.get("value")
        if v is None or n_tok not in tuple(v.shape[:2]) or kwargs.get("reference_points") is None:
            return None
        r = kwargs["reference_points"]
        kwargs = dict(kwargs, reference_points=w.u(r.float() * FINE).div(w.G).to(r.dtype))
        return args, kwargs
    n = 0
    for m in head.bev_modeling.head.transformer.decoder.modules():
        if isinstance(m, CustomMSDeformableAttention) and not getattr(m, "_warp_hooked", False):
            m.register_forward_pre_hook(hook, with_kwargs=True)
            m._warp_hooked = True
            n += 1
    return n


def _offset_class():
    from qwen_drive_perception import attention
    from qwen_drive_perception.attention import CustomMSDeformableAttention

    class OffsetWarpedMSDA(CustomMSDeformableAttention):
        """CustomMSDeformableAttention.forward with the sampling locations computed in the fine cells and
        then warped (Warp.locations), so the learned offsets keep their metric extent in the 1.024 m ring (the pre-hook
        warps only the reference points: offsets in cells then reach 2x as far in metres there). Shape-gated as the
        hook (G x G value); otherwise the upstream forward."""

        def forward(self, query, key=None, value=None, identity=None, query_pos=None, reference_points=None,
                    spatial_shapes=None, level_start_index=None, **kwargs):
            w = self._warp_offsets
            v = query if value is None else value
            if reference_points is None or w.G * w.G not in tuple(v.shape[:2]):
                return super().forward(query, key, value, identity, query_pos, reference_points, spatial_shapes,
                                       level_start_index, **kwargs)
            assert self.num_levels == 1                                         # no .tolist(): CUDA-graph capture
            self._warp_calls = getattr(self, "_warp_calls", 0) + 1
            if value is None:
                value = query
            if identity is None:
                identity = query
            if query_pos is not None:
                query = query + query_pos
            if not self.batch_first:
                query = query.permute(1, 0, 2)
                value = value.permute(1, 0, 2)
            bs, num_query, _ = query.shape
            _, num_value, _ = value.shape
            value = self.value_proj(value)
            value = value.view(bs, num_value, self.num_heads, -1)
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads,
                                                                   self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1).view(bs, num_query, self.num_heads, self.num_levels,
                                                                   self.num_points)
            if w.kc:
                sampling_locations = w.locations(reference_points[:, :, None, :, None, :], sampling_offsets).to(
                    torch.promote_types(reference_points.dtype, sampling_offsets.dtype))
            else:  # identity: upstream's arithmetic order (same value, bit-exact; the GPU identity check tests the rest
                # of this forward, the CPU self-test the warped locations)
                offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[
                    None, None, None, :, None, :]
            output = attention.multi_scale_deformable_attn_cuda(                # looked up per call: head_grad
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=64)
            output = self.output_proj(output)
            if not self.batch_first:
                output = output.permute(1, 0, 2)
            return self.dropout(output) + identity
    return OffsetWarpedMSDA


def enable_offsets(head) -> int:
    """, after install / Switch (and before det_sample / SamplingDecoder copy the decoder layers): the decoder's
    hooked BEV cross-attentions compute warped sampling locations exactly (OffsetWarpedMSDA). Returns the count."""
    w = head.bev_modeling.head.transformer._warp_occ
    cls = _offset_class()
    n = 0
    for m in head.bev_modeling.head.transformer.decoder.modules():
        if getattr(m, "_warp_hooked", False):
            m.__class__, m._warp_offsets = cls, w
            n += 1
    assert n, "warp_offsets: no hooked decoder attention (install warp_grid first)"
    return n


def _pooled(vt, w: Warp, j: torch.Tensor, mlvl_feats, img_depth, img_metas, out=None, scale=None, dense=False):
    """merged_pool.merged_pooled_volume onto the warped grid: [B, 1, G, G, Z, C] fp16 (cell sums; ``scale``:
    times cell_norm inside the kernel, ``dense``: no zero fill)."""
    from qdistill.merged_pool import merged_pool, merged_prep

    for m in img_metas:
        m["lidar2img"] = vt._format_lidar2img(m["lidar2img"][None, ...])
    voxel_coords, mask = vt.coord_preparing(img_metas)                      # fine indices, fine-grid mask
    vc = voxel_coords.clone()
    vc[..., 1:3] = j[voxel_coords[..., 1:3].long().clamp_(0, FINE - 1)].to(vc.dtype)  # one gather (same values)
    img_feats, depth = mlvl_feats[0], img_depth[0]
    B, Z = len(img_feats), vt.voxel_shape[2]
    return merged_pool(img_feats, depth, merged_prep(vc, mask, B, w.G, w.G, Z), B, w.G, w.G, Z, True, out, scale, dense)


def fold_pool(head) -> dict:
    """, after zhalf2 (head_chain "pool_fold"): the block's input scaling s * cell_norm moves into the pooling
    kernel and the kernel writes every voxel (no zero fill). Bit-exact (merged_pool_cuda.cu)."""
    vt = head.bev_modeling.view_trans
    assert hasattr(vt, "_zhalf_engine") and hasattr(vt, "_zhalf_up") and hasattr(vt, "_warp"), \
        "pool_fold: after warp_grid + zhalf2 on the warped grid"
    w, j, eng, up = vt._warp, vt._warp_j, vt._zhalf_engine, vt._zhalf_up
    n = w.cell_norm("cuda").to(torch.float16)                                  # [G, G], as the block's s * n
    vt._pool = lambda mlvl_feats, img_depth, img_metas, out=None: _pooled(vt, w, j, mlvl_feats, img_depth, img_metas,
                                                                             out=out, scale=n, dense=True)
    vt._block = lambda s: up(eng(s))                                           # plain callables
    return {"pool_fold": "cell_norm in the kernel, dense write"}


# ---------------------------------------------------------------------------------------------- deployment install
def install(head, b: float, voxel_engine: str | Path, ckpt: str | Path | None = None) -> dict:
    """In place, permanent; after merged_pool.install for the pooling (without it, e.g. trt_bev_encoder export, only
    the grid state / read-side patches / encoder size are set); before enc_trt. ``ckpt``: final.pt["warp_grid"]
    (trained BEV queries)."""
    from qdistill import trt_encoder
    from qdistill.voxel_engine import _Voxel

    w = Warp(b)
    q = None
    if ckpt:
        wg = torch.load(ckpt, map_location="cpu", weights_only=False)["warp_grid"]
        assert abs(wg["b"] - b) < 1e-6, (wg["b"], b)
        q = wg["bev_embedding"]
    _apply(head, warp_state(head, w, q))
    _patch_occ(head, w)
    n_hooks = _patch_decoder(head, w)
    trt_encoder.H = trt_encoder.W = w.G
    vt = head.bev_modeling.view_trans
    pool = "deferred (no merged_pool)"
    if hasattr(vt, "_pool") and hasattr(vt, "_block"):
        eng = _Voxel(voxel_engine, w.G, w.G, vt.voxel_shape[2])
        vt._warp_engine = eng
        j = w.index("cuda")
        vt._warp, vt._warp_j = w, j                                            # fold_pool
        n = w.cell_norm("cuda").to(torch.float16)[None, None, :, :, None, None]
        vt._pool = lambda mlvl_feats, img_depth, img_metas, out=None: _pooled(vt, w, j, mlvl_feats, img_depth,
                                                                                 img_metas, out=out)
        vt._block = lambda s: eng(s * n)                                       # plain callables

        def forward(mlvl_feats, img_depth, img_metas):
            return vt._block(vt._pool(mlvl_feats, img_depth, img_metas))
        vt.forward = forward
        pool = "warped"
    torch.cuda.empty_cache()
    return {"warp_grid": b, "grid": w.G, "pool": pool, "decoder_hooks": n_hooks, "voxel_engine": str(voxel_engine),
            "queries": str(ckpt) if ckpt else "pooled"}


# ---------------------------------------------------------------------------------------------------- training
class Switch:
    """The warped grid in training (train_enc_recover --warp-grid, train_det_sampling): the head
    keeps its 200 x 200 state for the Stage-2 targets; ``with sw.half():`` swaps in the warped state. Read-side patches
    are installed for good (shape-gated). ``queries``: the warped BEV embedding as an fp32 Parameter."""

    def __init__(self, head, voxel_engine, b: float = 25.6, z8_engine=None):
        """``z8_engine``: a zhalf v2 engine (16-level warped input, 16 -> 8 compression inside, 8 levels out)
        in place of ``voxel_engine``; its output goes back to 16 levels (zhalf.zup) for prior / p16."""
        from qdistill.voxel_engine import _Voxel

        self.head, self.w, self.b, self.voxel_engine = head, Warp(b), b, str(voxel_engine)
        self.z8_engine = str(z8_engine) if z8_engine else None
        self.full_st = _state(head)
        self.warp_st = warp_state(head, self.w)
        self.queries = nn.Parameter(self.warp_st["bev_embedding"].weight.detach().float().clone())
        _patch_occ(head, self.w)
        self.n_hooks = _patch_decoder(head, self.w)
        vt = head.bev_modeling.view_trans
        if self.z8_engine:
            from qdistill import zhalf
            eng = _Voxel(self.z8_engine, self.w.G, self.w.G, vt.voxel_shape[2], z_out=zhalf.Z8)
            self.eng = lambda s: zhalf.zup(eng(s))
        else:
            self.eng = _Voxel(voxel_engine, self.w.G, self.w.G, vt.voxel_shape[2])
        self.j = self.w.index("cuda")
        self.n = self.w.cell_norm("cuda").to(torch.float16)[None, None, :, :, None, None]
        self.a = self.w.avg("cuda")
        self.keep_volume = False                                                # to_half keeps c["y"]

    @contextmanager
    def half(self):
        fc = self.head.bev_modeling.head.feat_cropper
        self.full_st["_map_grid"] = fc._map_grid
        _apply(self.head, self.warp_st)
        try:
            yield
        finally:
            _apply(self.head, self.full_st)

    def retarget(self, tail):
        """A bev2d.EncoderTail built at 200 x 200 (after its self-check) -> the warped grid."""
        with self.half(), torch.no_grad():
            hd = self.head.bev_modeling.head
            tail.bev_h, tail.bev_w = hd.bev_h, hd.bev_w
            tail.grid_length = (hd.real_h / hd.bev_h, hd.real_w / hd.bev_w)   # unused by the encoder
            mask = torch.zeros((1, hd.bev_h, hd.bev_w), device="cuda", dtype=torch.bfloat16)
            pos = hd.positional_encoding(mask).to(torch.bfloat16)
            tail.bev_pos = pos.flatten(2).permute(2, 0, 1).contiguous()
        return tail

    def volume(self, c, metas):
        m = dict(metas)
        l2i = m["lidar2img"]
        if l2i.ndim == 5:                                                       # formatted in place by the targets' run
            m["lidar2img"] = l2i[0, :, 0]
        with torch.no_grad():
            s = _pooled(self.head.bev_modeling.view_trans, self.w, self.j, [c["vit"][None]], [c["depth"][None]], [m])
            return self.eng(s * self.n)

    def to_half(self, c: dict, metas, prior_fn) -> dict:
        """Stage-2 capture -> the student path's inputs on the warped grid, in place: prior / p16 from the warped block,
        the bev_embed target warped (cell means)."""
        y = self.volume(c, metas)
        with torch.no_grad():
            c["prior"] = prior_fn(y)                                            # [G * G, 256]
            c["p16"] = self.head.bev_modeling.head.transformer.uvtr_occ_proj(y).flatten(1, 2)
            c["bev"] = warp_bev(c["bev"], self.a)
        if self.keep_volume:
            c["y"] = y                                                          # [1, 256, 16, G, G] (zup'd)
        return c

    def state(self) -> dict:
        return {"b": self.b, "voxel_engine": self.voxel_engine, "bev_embedding": self.queries.detach().cpu(),
                **({"z8_engine": self.z8_engine} if self.z8_engine else {})}


# ------------------------------------------------------------------------------------------------------- CLIs
def self_test() -> dict:
    r = {}
    for b, G in ((25.6, 150), (20.48, 140), (51.2, 200)):
        w = Warp(b)
        assert w.G == G, (b, w.G)
        j = w.index()
        assert j.min() == 0 and j.max() == G - 1 and bool((j[1:] - j[:-1] >= 0).all()), b
        t = torch.linspace(0, FINE, 4001, dtype=torch.float64)
        u = w.u(t)
        assert bool((u[1:] - u[:-1] >= 0).all()) and abs(float(u[0])) < 1e-9 and abs(float(u[-1]) - G) < 1e-9, b
        tc = torch.arange(FINE, dtype=torch.float64) + 0.5                     # fine cell centres -> their warped cell
        assert torch.equal(torch.floor(w.u(tc)).long(), j), b
        a = w.avg()
        assert torch.allclose(a.sum(1), torch.ones(G)), b
        r[str(b)] = {"G": G, "coarse_per_side": w.kc, "fine": 2 * w.nf, "cells_vs_d053": round(G * G / FINE ** 2, 3)}
    # identity (b = 51.2): map grid = heads.BevFeatureSlicer._grid, occupancy x / y = _axis_center_coords
    w = Warp(51.2)
    xs = torch.arange(-30.0 + 0.075, 30.0, 0.15)
    ref = xs / (HALF - CELL / 2)
    assert float((w.norm_ac(xs) - ref).abs().max()) < 1e-5, "map grid identity"
    tc = torch.arange(200, dtype=torch.float32)                                # occupancy fine normalised -> warped
    n_f = 2.0 * tc / (FINE - 1) - 1.0
    n_w = 2.0 * (w.u((n_f + 1.0) * (FINE - 1) / 2.0 + 0.5) - 0.5) / (w.G - 1) - 1.0
    assert float((n_w - n_f).abs().max()) < 1e-5, "occ grid identity"
    # 25.6: the warp leaves the fine zone at 0.512 m and is continuous at the knots
    w = Warp(25.6)
    k = torch.tensor([w.t0 - 1e-6, w.t0 + 1e-6, w.t1 - 1e-6, w.t1 + 1e-6], dtype=torch.float64)
    uk = w.u(k)
    assert abs(float(uk[1] - uk[0])) < 1e-5 and abs(float(uk[3] - uk[2])) < 1e-5, "continuity"
    r["knots_25.6_m"] = [round(float(x), 3) for x in w.u(torch.tensor([float(w.t0), float(w.t1)]))]
    # offsets: u_inv round trip; warped sampling points land on the metric points (fine zone, ring, across
    # the knots); identity = the reference + offset / 200
    g = torch.Generator().manual_seed(0)
    t = torch.rand(10000, generator=g, dtype=torch.float64) * FINE
    assert float((w.u_inv(w.u(t)) - t).abs().max()) < 1e-9, "u_inv"
    ref = torch.rand(10000, 2, generator=g, dtype=torch.float64)
    off = (torch.rand(10000, 2, generator=g, dtype=torch.float64) - 0.5) * 16              # +-8 fine cells
    loc = w.locations(ref, off).double()
    assert float((w.u_inv(loc * w.G) - (ref * FINE + off)).abs().max()) < 1e-3, "offsets: metric point"
    big = off.abs() > 1                                                                     # fp32 locations / off
    ring = ((ref * FINE < w.t0 - 8) | (ref * FINE > w.t1 + 8)) & big
    fine = (ref * FINE > w.t0 + 8) & (ref * FINE < w.t1 - 8) & big
    step = (loc - w.locations(ref, torch.zeros_like(off)).double()) * w.G / off             # warped cells per offset
    assert float((step[ring] - 0.5).abs().max()) < 1e-3 and float((step[fine] - 1).abs().max()) < 1e-3, "offset scale"
    wi = Warp(51.2)
    assert float((wi.locations(ref, off).double() - (ref + off / FINE)).abs().max()) < 1e-6, "offsets identity"
    r["offsets_scale"] = {"fine": 1.0, "ring": 0.5}
    return r


def check(args) -> dict:
    """GPU: the same frames through the chain (``--chain``) and through the chain + warp_grid at --b, compared (b 51.2
    must reproduce the chain: occ / map label equality, det max abs diff)."""
    import copy
    import gc

    sys.path.insert(0, str(ROOT / "scripts/profiling"))
    from e2e_latency import compare
    from qdistill.student_io import load_student
    from qdistill import head_chain, merged_pool
    from qdistill.data import load_frame_images, split_tokens
    from qdistill.teacher import load_head, load_processor, run_head
    from qdistill.trt_install import TrtHead
    from qwen_drive_perception.dataset import PerceptionFrame

    student, norm = load_student(args.ckpt, args.cache)
    processor = load_processor(args.weights)
    tokens = split_tokens(args.splits, "val_quick")[:: args.stride][: args.frames]
    taps = []
    for t in tokens:
        images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        taps.append((t, norm.denormalize("vit", vit_n[0]).cpu(), norm.denormalize("llm", llm_n[0]).cpu()))
    del student
    res = {}
    wx = f"warp_grid:{args.b}:{args.voxel_engine}" + (",warp_offsets" if args.offsets else "")
    calls = None
    for arm, extra in (("ref", ""), ("warp", wx)):
        head = load_head(args.weights)
        trt_head = TrtHead(head, args.base_voxel_engine, args.refiner_engine)
        merged_pool.install(head, trt_head)
        items = [s for s in args.chain.split(",") if s.strip()]
        k = next((i for i, s in enumerate(items) if s.startswith("enc_trt:")), len(items))
        spec = ",".join(items[:k] + ([extra] if extra else []) + items[k:])
        if args.chain_b:                                                       # two given chains, no warp
            spec = args.chain if arm == "ref" else args.chain_b
        print(f"{arm}: {head_chain.install(head, spec)}", file=sys.stderr, flush=True)
        out = {}
        for t, vit, llm in taps:
            metas = processor(PerceptionFrame(args.packed / t))[1]
            with torch.no_grad():
                o = run_head(head, vit.cuda(), llm.cuda(), copy.deepcopy(metas))
            out[t] = {"occ": o["occ"], "map": o["map"], "det": [o["boxes"], o["scores"], o["labels"]]}
        res[arm] = out
        if arm == "warp" and (args.offsets or args.chain_b):                  # the offset path really ran
            calls = sum(getattr(m, "_warp_calls", 0) for m in head.modules())
        del head, trt_head
        gc.collect()
        torch.cuda.empty_cache()
    r = compare(res["ref"], res["warp"])
    if args.offsets or args.chain_b:
        r["offset_forward_calls"] = calls
    return r


def block_engine(args) -> dict:
    from qdistill.trt_utils import build_engine
    w = Warp(args.b)
    return build_engine(BLOCK_ONNX, "s", (1, 1, w.G, w.G, 16, 256), args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("self-test", "check", "block-engine"))
    p.add_argument("--b", type=float, default=25.6)
    p.add_argument("--out", type=Path)
    p.add_argument("--chain", default="")
    p.add_argument("--voxel-engine", type=Path, help="check: the G x G block engine for the warp arm")
    p.add_argument("--offsets", action="store_true", help="check: + warp_offsets")
    p.add_argument("--chain-b", default="", help="check: compare --chain with this chain instead")
    p.add_argument("--base-voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--frames", type=int, default=25)
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = p.parse_args()
    r = {"self-test": lambda: self_test(), "check": lambda: check(args), "block-engine": lambda: block_engine(args)}[args.cmd]()
    print(json.dumps(r), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
