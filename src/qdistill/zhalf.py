"""the voxel block at 8 height levels instead of 16, everything else unchanged.

    python -m qdistill.zhalf build --ckpt runs/<run>/best.pt --out outputs/e106/<name>_fp16.engine   # Z 8 engine
    python -m qdistill.zhalf self-test                                                                # CPU checks

Pooling stays at 16 levels (unchanged geometry); the block reads the mean of each pair of levels (0.8 m cells instead of
0.4 m) and its output is interpolated back to 16 levels, linearly at the cell centres (align_corners=False, edges
clamped), so uvtr_query_proj (256 x 16 channels), the occupancy volume adapter and every later module see the shapes
they were trained on. the r64 block at Z 8 is 27.3 ms vs 52.6 at Z 16 (TensorRT fp16, 200 x 200).

  ZHalfBlock(block)          training wrapper (distill_voxel_block --zhalf): [B, 1, X, Y, 16, C] -> [B, C, 16, Y, X]
  install(head, engine)      deployment, after merged_pool (head_chain zhalf:<engine>): vt._block -> zup(eng(pair_mean))
  Switch(head, engine)       train_enc_recover / train_det_sampling --zhalf: prior / p16 from the Z 8 block (
                             Switch.to_block)

the 16 -> 8 compression inside the block engine (CompressedZ8Block, e.g. stride6 = the first conv strided in
Z, initialised exactly from the block by from_zhalf), so no eager pair_mean:
    python -m qdistill.zhalf build2 --ckpt runs/<run>/best.pt --out <engine> --grid 150   # 16 levels in, 8 out
  install2(head, engine)     head_chain zhalf2:<engine> (after warp_grid): vt._block -> zup(eng(s x cell_norm))
  warp_grid.Switch(..., z8_engine=<engine>)   training on the warped grid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
X = Y = 200
Z16, Z8, C = 16, 8, 256


def pair_mean(s: torch.Tensor) -> torch.Tensor:
    """Pooled volume [B, n, X, Y, 16, C] -> [B, n, X, Y, 8, C]: mean of levels (2k, 2k + 1)."""
    b, n, x, y, z, c = s.shape
    return s.view(b, n, x, y, z // 2, 2, c).mean(5).contiguous()


def zup(y: torch.Tensor) -> torch.Tensor:
    """Block output [B, C, 8, Y, X] -> [B, C, 16, Y, X], linear at cell centres: level 2k = 0.75 y[k] + 0.25 y[k - 1],
    2k + 1 = 0.75 y[k] + 0.25 y[k + 1], edges clamped (= F.interpolate linear, align_corners=False, along Z)."""
    lo = torch.cat([y[:, :, :1], y[:, :, :-1]], 2)
    hi = torch.cat([y[:, :, 1:], y[:, :, -1:]], 2)
    out = torch.stack([0.75 * y + 0.25 * lo, 0.75 * y + 0.25 * hi], 3)          # [B, C, 8, 2, Y, X]
    b, c, z, _, h, w = out.shape
    return out.view(b, c, 2 * z, h, w)


class ZHalfBlock(nn.Module):
    """A voxel block (light_blocks) run at 8 levels inside the 16-level interface."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.cfg = {**block.cfg, "zhalf": True}

    def forward(self, s: torch.Tensor, return_layers: bool = False):
        assert not return_layers, "zhalf: no per-layer outputs"
        return zup(self.block(pair_mean(s)))


def init_stride(w: torch.Tensor, k: int) -> torch.Tensor:
    """The 3x3x3 first conv at Z 8 on pair means [O, I, 3, 3, 3] -> a 3x3xk conv, Z stride 2, on the 16 levels.
    Output level j of the Z 8 conv reads pairs j - 1, j, j + 1 = fine levels 2j - 2 .. 2j + 3; the strided conv with
    padding (k - 2) / 2 reads 2j + 1 - k / 2 .. 2j + k / 2. k 6: each tap split over its pair (exact, zero padding
    included). k 4: the outer pairs reach only one fine level each, which takes the pair's whole tap (exact when the
    two levels of an outer pair are equal; the centre pair stays exact)."""
    w0, w1, w2 = w.unbind(4)
    taps = [w0, w0, w1, w1, w2, w2] if k == 6 else [2 * w0, w1, w1, 2 * w2]
    assert k in (4, 6), k
    return torch.stack(taps, 4) / 2


class CompressedZ8Block(nn.Module):
    """the 16 -> 8 height compression inside the block (one engine, no eager pair_mean), then an r64 block at
    8 levels; [B, 1, X, Y, 16, C] -> [B, C, 8, Y, X] (zup is folded into the consumers, not applied here).
    ``mode``: "mean" fixed pair-mean; "dw" depthwise 1x1x2 stride-2 conv (per-channel pair
    weights + bias); "full" 1x1x2 stride-2 conv C -> C; "stride<k>" the block's first 3x3x3 conv replaced by a 3x3xk conv
    with Z stride 2. dw / full start as the exact
    pair-mean; stride init: init_stride (from the block's own 3x3x3 first conv)."""

    def __init__(self, block: nn.Module, mode: str = "dw", channels: int = C):
        super().__init__()
        self.mode, self.block = mode, block
        self.cfg = {**block.cfg, "z8_compress": mode}
        self.comp = None
        if mode in ("dw", "full"):
            g = channels if mode == "dw" else 1
            self.comp = nn.Conv3d(channels, channels, (1, 1, 2), stride=(1, 1, 2), groups=g)
            with torch.no_grad():
                w = torch.zeros_like(self.comp.weight)                      # [C, C / g, 1, 1, 2]
                if mode == "dw":
                    w[:, 0] = 0.5
                else:
                    w[torch.arange(channels), torch.arange(channels)] = 0.5
                self.comp.weight.copy_(w)
                self.comp.bias.zero_()
        elif mode.startswith("stride"):
            k = int(mode.removeprefix("stride"))
            first = block.spatial[0]
            block.spatial[0] = nn.Conv3d(first.in_channels, first.out_channels, (3, 3, k), stride=(1, 1, 2),
                                         padding=(1, 1, (k - 2) // 2), bias=first.bias is not None)
            with torch.no_grad():
                block.spatial[0].weight.copy_(init_stride(first.weight, k))
                if first.bias is not None:
                    block.spatial[0].bias.copy_(first.bias)
        else:
            assert mode == "mean", mode

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return self.block(pair_mean(s))
        if self.comp is None:                                                   # strided first conv reads 16 levels
            return self.block(s)
        x = self.comp(s[:, 0].permute(0, 4, 1, 2, 3))                          # [B, C, X, Y, 8]
        return self.block(x.permute(0, 2, 3, 4, 1)[:, None])                   # back to [B, 1, X, Y, 8, C]




def export_onnx(block: nn.Module, path: Path):
    """The inner block (dynamic X / Y / Z, as trt_voxel_block)."""
    block = block.float().eval().cpu()
    torch.onnx.export(block, (torch.randn(1, 1, 8, 8, 4, C),), str(path), input_names=["s"], output_names=["y"],
                      opset_version=17, dynamic_axes={"s": {2: "X", 3: "Y", 4: "Z"}, "y": {2: "Z", 3: "Y", 4: "X"}},
                      dynamo=False)


def export_onnx_v2(block: nn.Module, path: Path):
    """CompressedZ8Block: 16 levels in (fixed), 8 out; dynamic X / Y."""
    block = block.float().eval().cpu()
    torch.onnx.export(block, (torch.randn(1, 1, 8, 8, Z16, C),), str(path), input_names=["s"], output_names=["y"],
                      opset_version=17, dynamic_axes={"s": {2: "X", 3: "Y"}, "y": {3: "Y", 4: "X"}}, dynamo=False)


def load_block(ckpt: str | Path) -> nn.Module:
    """best.pt of distill_voxel_block (zhalf or not) -> the inner block (fp32, CPU); a v2 checkpoint (cfg
    z8_compress) -> the whole CompressedZ8Block."""
    from qdistill.light_blocks import build_voxel_block

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    if cfg.get("z8_compress"):
        cfg.pop("zhalf", None)
        blk = build_voxel_block(cfg)
        blk.load_state_dict(ck["model"])
        return blk.float().eval()
    z = cfg.pop("zhalf", False)
    inner = build_voxel_block(cfg)
    sd = ck["model"]
    if z:
        sd = {k.removeprefix("block."): v for k, v in sd.items()}
    inner.load_state_dict(sd)
    return inner.float().eval()


def build(ckpt: str | Path, out: str | Path, workspace_gb: int = 2, grid: int = X) -> dict:
    from qdistill.trt_utils import build_engine

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    block = load_block(ckpt)
    onnx_path = out.with_suffix(".onnx")
    export_onnx(block, onnx_path)
    r = build_engine(onnx_path, "s", (1, 1, grid, grid, Z8, C), out, "fp16", workspace_gb=workspace_gb)
    assert not r.get("failed"), r
    # engine vs the PyTorch block (fp32) on a random non-negative volume at the pooled scale
    from qdistill.voxel_engine import _Voxel

    torch.manual_seed(0)
    s = torch.relu(torch.randn(1, 1, grid, grid, Z8, C, device="cuda")) * 0.5
    with torch.no_grad():
        ref = block.cuda()(s).float()
        got = _Voxel(out, grid, grid, Z8)(s.half()).float()
    r["trt_vs_fp32_nmse"] = float(((got - ref) ** 2).mean() / ref.var())
    return r


def build2(ckpt: str | Path, out: str | Path, workspace_gb: int = 2, grid: int = X, mode: str = "stride6") -> dict:
    """v2 engine [1, 1, G, G, 16, C] -> [1, C, 8, G, G]. ``ckpt``: a v2 best.pt, or (identity / init engine) a v1 zhalf
    best.pt wrapped with ``mode`` (from_zhalf)."""
    from qdistill.voxel_engine import _Voxel
    from qdistill.trt_utils import build_engine

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    block = load_block(ckpt)
    # light_blocks builds qdistill.zhalf's class, not __main__'s (python -m qdistill.zhalf build2)
    from qdistill.zhalf import CompressedZ8Block as Z8Cls
    if not isinstance(block, Z8Cls):
        block = Z8Cls(block, mode).float().eval()
    onnx_path = out.with_suffix(".onnx")
    export_onnx_v2(block, onnx_path)
    r = build_engine(onnx_path, "s", (1, 1, grid, grid, Z16, C), out, "fp16", workspace_gb=workspace_gb)
    assert not r.get("failed"), r
    torch.manual_seed(0)
    s = torch.relu(torch.randn(1, 1, grid, grid, Z16, C, device="cuda")) * 0.5
    with torch.no_grad():
        ref = block.cuda()(s).float()
        got = _Voxel(out, grid, grid, Z16, z_out=Z8)(s.half()).float()
    r["trt_vs_fp32_nmse"] = float(((got - ref) ** 2).mean() / ref.var())
    r["mode"] = block.mode
    return r


def zup_matrix() -> torch.Tensor:
    """[16, 8] U with zup(y)[:, :, z] = sum_k U[z, k] y[:, :, k] (rows sum to 1)."""
    return zup(torch.eye(Z8).view(1, Z8, Z8, 1, 1))[0, :, :, 0, 0].T.contiguous()


def install_fold(head) -> dict:
    """L2: the consumers read the block's 8-level output directly, exactly (zup is linear, rows summing to 1):
      prior   uvtr_query_proj (1x1 Conv2d over C x 16 channels, index c * 16 + z) -> W'[o, c, k] = sum_z W[o, c, z] U[z, k]
              on C x 8 channels (half the GEMM, no 16-level volume);
      occ     upstream grid_sample(V16) then uvtr_occ_proj (1x1x1, C -> 16): instead zup(uvtr_occ_proj(V8)) then the
              grid_sample on 16 channels (the projection commutes with zup and with the crop, whose bilinear weights sum
              to 1 inside the grid, lift2d), and the fuse skips its projection for that input.
    Shape-gated (an 8-level C-channel volume / a middle_dims-channel occupancy feature); else upstream."""
    bev = head.bev_modeling
    tr = bev.head.transformer
    U = zup_matrix()
    qp = bev.uvtr_query_proj
    base = getattr(qp, "m", qp)  # a trained projection is a bev2d.Fp16Run over its fp32 master (heads_ckpt)
    w = base.weight.detach().float()                                            # [O, C * 16, 1, 1]
    o, ck = w.shape[:2]
    c = ck // Z16
    wf = torch.einsum("ocz,zk->ock", w.view(o, c, Z16).cpu(), U).reshape(o, c * Z8, 1, 1)
    fold = nn.Conv2d(c * Z8, o, 1).to(device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        fold.weight.copy_(wf.to(fold.weight))
        fold.bias.copy_(base.bias)
    fold.requires_grad_(False)
    if base is not qp:                                                          # folded from the fp32 master, run fp16
        from qdistill.bev2d import Fp16Run
        fold = Fp16Run(fold)
    bev._uvtr_query_proj_z8 = fold
    f_tok = bev._uvtr_voxel_to_bev_tokens

    def tokens(v):
        if v.shape[2] != Z8:
            return f_tok(v)
        b, ch, d, h, wd = v.shape
        q = bev._uvtr_query_proj_z8(v.reshape(b, ch * d, h, wd))
        if q.shape[0] > 1:
            q = q.mean(dim=0, keepdim=True)
        return q.squeeze(0).permute(1, 2, 0).reshape(-1, q.shape[1]).contiguous()
    bev._uvtr_voxel_to_bev_tokens = tokens
    f_adapt, f_fuse = tr._adapt_volume_for_occ, tr._fuse_uvtr_occ_feat
    mid = tr.middle_dims

    def adapt(volume_feat, bev_h, bev_w, occ_pc_range, occ_voxel_size):
        if volume_feat.shape[2] == Z8 and volume_feat.shape[1] != mid:
            volume_feat = zup(tr.uvtr_occ_proj(volume_feat))                    # [B, 16 ch, 16, Y, X]
        return f_adapt(volume_feat, bev_h, bev_w, occ_pc_range, occ_voxel_size)

    def fuse(bev_feat, uvtr_occ_feat):
        if uvtr_occ_feat.shape[1] == mid and mid != tr.embed_dims:              # already projected (adapt above)
            return bev_feat + tr.uvtr_occ_fuse(torch.cat([bev_feat, uvtr_occ_feat.to(bev_feat.dtype)], dim=1))
        return f_fuse(bev_feat, uvtr_occ_feat)
    tr._adapt_volume_for_occ, tr._fuse_uvtr_occ_feat = adapt, fuse
    return {"fold": "uvtr_query_proj C x 8, uvtr_occ_proj before zup"}


def install2(head, voxel_engine: str | Path, fold: bool = False) -> dict:
    """after merged_pool.install (and warp_grid, if any): the block becomes the v2 engine on the 16-level pooled
    volume (x the warp's cell_norm), output back to 16 levels by zup; with ``fold`` (L2) the output stays at 8 levels
    and the consumers are folded (install_fold)."""
    from qdistill.voxel_engine import _Voxel

    vt = head.bev_modeling.view_trans
    assert hasattr(vt, "_pool") and hasattr(vt, "_block"), "zhalf2 needs merged_pool.install first (--merged-pool)"
    w = getattr(head.bev_modeling.head.transformer, "_warp_occ", None)
    if w is not None:
        assert hasattr(vt, "_warp_engine"), "zhalf2 after warp_grid: warp_grid must have installed its pooling"
        eng = _Voxel(voxel_engine, w.G, w.G, Z16, z_out=Z8)
        n = w.cell_norm("cuda").to(torch.float16)[None, None, :, :, None, None]
        up = (lambda y: y) if fold else zup
        vt._block = lambda s: up(eng(s * n))                                     # a plain callable
        vt._zhalf_up = up                                                        # warp_grid.fold_pool
    else:
        eng = _Voxel(voxel_engine, X, Y, Z16, z_out=Z8)
        up = (lambda y: y) if fold else zup
        vt._block = lambda s: up(eng(s))
    vt._zhalf_engine = eng
    r = {"zhalf2": str(voxel_engine), "levels": Z8, "grid": w.G if w is not None else X}
    if fold:
        r.update(install_fold(head))
    return r


class Switch:
    """train_enc_recover / train_det_sampling: replace the Stage-2 capture's prior / p16 with the Z 8 chain's (pooling
    unchanged, block engine on the pair means, output back to 16 levels)."""

    def __init__(self, head, voxel_engine: str | Path):
        from qdistill.voxel_engine import _Voxel

        self.head = head
        self.eng = _Voxel(voxel_engine, X, Y, Z8)

    def volume(self, c, metas):
        from qdistill import merged_pool

        m = dict(metas)
        l2i = m["lidar2img"]
        if l2i.ndim == 5:                                                       # formatted in place by the targets' run
            m["lidar2img"] = l2i[0, :, 0]
        with torch.no_grad():
            s = merged_pool.merged_pooled_volume(self.head.bev_modeling.view_trans, [c["vit"][None]],
                                                 [c["depth"][None]], [m])
            return zup(self.eng(pair_mean(s)))

    def to_block(self, c: dict, metas, prior_fn) -> dict:
        y = self.volume(c, metas)
        with torch.no_grad():
            c["prior"] = prior_fn(y)                                            # [200 * 200, 256]
            c["p16"] = self.head.bev_modeling.head.transformer.uvtr_occ_proj(y).flatten(1, 2)
        return c


def install(head, voxel_engine: str | Path) -> dict:
    """After merged_pool.install (vt._pool / vt._block split): the block becomes the Z 8 engine on the pair means.
    after warp_grid.install too (warp_grid first), the engine is G x G and its input the warped cell sums
    x cell_norm, as warp_grid's own block (the pooling stays warp_grid's)."""
    from qdistill.voxel_engine import _Voxel

    vt = head.bev_modeling.view_trans
    assert hasattr(vt, "_pool") and hasattr(vt, "_block"), "zhalf needs merged_pool.install first (--merged-pool)"
    w = getattr(head.bev_modeling.head.transformer, "_warp_occ", None)
    if w is not None:
        assert hasattr(vt, "_warp_engine"), "zhalf after warp_grid: warp_grid must have installed its pooling"
        eng = _Voxel(voxel_engine, w.G, w.G, Z8)
        n = w.cell_norm("cuda").to(torch.float16)[None, None, :, :, None, None]
        vt._block = lambda s: zup(eng(pair_mean(s * n)))                         # a plain callable
    else:
        eng = _Voxel(voxel_engine, X, Y, Z8)
        vt._block = lambda s: zup(eng(pair_mean(s)))
    vt._zhalf_engine = eng
    return {"zhalf": str(voxel_engine), "levels": Z8, "grid": w.G if w is not None else X}


def self_test():
    import torch.nn.functional as F

    torch.manual_seed(0)
    y = torch.randn(2, 3, 8, 5, 4)
    ref = F.interpolate(y, size=(16, 5, 4), mode="trilinear", align_corners=False)
    assert torch.allclose(zup(y), ref, atol=1e-6), (zup(y) - ref).abs().max()
    s = torch.randn(1, 1, 4, 3, 16, 2)
    pm = pair_mean(s)
    assert torch.allclose(pm[..., 3, :], (s[..., 6, :] + s[..., 7, :]) / 2)
    from qdistill.light_blocks import TeacherShapedVoxelEncoder

    zb = ZHalfBlock(TeacherShapedVoxelEncoder(8, rank=4))
    o = zb(torch.randn(1, 1, 6, 5, 16, 8))
    assert o.shape == (1, 8, 16, 5, 6), o.shape
    import copy
    inner = TeacherShapedVoxelEncoder(8, rank=4).eval()
    s = torch.randn(1, 1, 6, 5, 16, 8)
    with torch.no_grad():
        ref = inner(pair_mean(s))
        st6 = CompressedZ8Block(copy.deepcopy(inner), "stride6")(s)
        flat = s.view(1, 1, 6, 5, 8, 2, 8)[..., :1, :].expand(-1, -1, -1, -1, -1, 2, -1).reshape(s.shape)
        st4 = CompressedZ8Block(copy.deepcopy(inner), "stride4")(flat)
        dw = CompressedZ8Block(copy.deepcopy(inner), "dw", channels=8)(s)
    assert torch.allclose(st6, ref, atol=1e-5), (st6 - ref).abs().max()
    assert torch.allclose(st4, inner(pair_mean(flat)), atol=1e-5)               # exact on pair-constant volumes
    assert torch.allclose(dw, ref, atol=1e-5)
    U = zup_matrix()                                                             # L2 fold of uvtr_query_proj
    y8 = torch.randn(1, 6, Z8, 4, 5)
    assert torch.allclose(torch.einsum("zk,bckyx->bczyx", U, y8), zup(y8), atol=1e-6) and torch.allclose(U.sum(1), torch.ones(Z16))
    qp = nn.Conv2d(6 * Z16, 7, 1)
    wf = torch.einsum("ocz,zk->ock", qp.weight.detach().view(7, 6, Z16), U).reshape(7, 6 * Z8, 1, 1)
    with torch.no_grad():
        ref = qp(zup(y8).reshape(1, 6 * Z16, 4, 5))
        got = F.conv2d(y8.reshape(1, 6 * Z8, 4, 5), wf, qp.bias)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()
    return {"zup_vs_interpolate": "ok", "pair_mean": "ok", "wrapper_shape": list(o.shape), "query_fold": "ok",
            "stride6_init_exact": "ok", "stride4_init_pair_constant": "ok", "dw_init_exact": "ok"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--grid", type=int, default=X, help="X = Y of the engine (150, the warped grid)")
    b2 = sub.add_parser("build2")
    b2.add_argument("--ckpt", required=True)
    b2.add_argument("--out", required=True)
    b2.add_argument("--grid", type=int, default=X)
    b2.add_argument("--mode", default="stride6", help="compression when --ckpt is a v1 block")
    sub.add_parser("self-test")
    a = p.parse_args()
    if a.cmd == "build":
        r = build(a.ckpt, a.out, grid=a.grid)
    elif a.cmd == "build2":
        r = build2(a.ckpt, a.out, grid=a.grid, mode=a.mode)
    else:
        r = self_test()
    print(json.dumps(r, default=str), flush=True)


if __name__ == "__main__":
    main()
