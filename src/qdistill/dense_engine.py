"""the dense heads as one TensorRT engine (~21 ms of the head's 87 ms GPU time is glue; occ_fuse
alone is 3.6 of 4.7 ms bf16 <-> fp16 casts).

One engine from the two tensors the occupancy / map branches read to their two outputs:
    bev [1, 256, G, G] (bev_feat_ego)   vol [1, 256, 8, G, G]
 -> occ [1, 200, 200, 16, 10] (occ_pred)   seg [1, 6, 200, 400] (seg_preds)
inside: occ crop of the BEV (view 16 ch x 16 Z) and of zup(uvtr_occ_proj(vol)), uvtr_occ_fuse + residual, the
2D refiner, permute, occ_pred_head; map crop (BevFeatureSlicer) + seg_decoder. Weights: the trained fp32 masters
(final.pt["heads"]) and the refiner's best.pt.

The occupancy crops are F.grid_sample on a fixed axis-aligned 5D grid (bilinear, align_corners=True, zeros padding),
which TensorRT cannot run (no 5D GridSample). The grid is separable (asserted), so each becomes a 4D grid_sample over
x / y with Z in the channels, then a [Zt, Zs] interpolation matrix over Z with zup (linear, Z 8 -> 16) folded in for
the volume (a grouped 1x1 conv over the Z channels): the same weights, exact up to fp16 accumulation. The map crop is a
4D grid_sample as upstream. crop="lerp": x / y as two gathers + weights per axis instead of grid_sample. (Smoke: x / y
as dense interpolation-matrix GEMMs took 3.9 / 5.5 ms in the engine, the Z matrix as a batched GEMM 5.9 ms.)

    DenseHeads.from_modules(defs, z_ch, heads_ckpt, refiner_ckpt, grids)   fp32 module (export / reference)
    capture_grids(head, metas)                                    the chain's occ / map grids
    install(head, engine)                                         head_chain "dense_trt:<engine>" (last item)
    python -m qdistill.dense_engine self-test                     CPU: separable crop == F.grid_sample
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from qdistill import trt_modules

NAMES = ("uvtr_occ_proj", "uvtr_occ_fuse", "occ_pred_head", "map_seg")  # trt_modules.TARGETS inside the engine


def axis_matrix(g: torch.Tensor, size: int) -> torch.Tensor:
    """1-D bilinear weights of grid_sample (align_corners=True, zeros padding): normalised coords g [T] -> M [T, S]."""
    p = (g.double() + 1) / 2 * (size - 1)
    i0 = torch.floor(p)
    w1 = p - i0
    m = torch.zeros(len(g), size, dtype=torch.float64)
    for off, w in ((0, 1 - w1), (1, w1)):
        i = (i0 + off).long()
        ok = (i >= 0) & (i < size)
        m[torch.arange(len(g))[ok], i[ok]] += w[ok]
    return m.float()


def lerp_taps(g: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """axis_matrix as gather taps: [2, T] source indices (clamped in range) and [2, T] weights (0 for a tap outside
    the source: zeros padding)."""
    p = (g.double() + 1) / 2 * (size - 1)
    i0 = torch.floor(p)
    w1 = p - i0
    idx, wts = [], []
    for off, w in ((0, 1 - w1), (1, w1)):
        i = (i0 + off).long()
        ok = (i >= 0) & (i < size)
        idx.append(i.clamp(0, size - 1))
        wts.append(torch.where(ok, w, torch.zeros_like(w)))
    return torch.stack(idx), torch.stack(wts).float()


def separable(grid: torch.Tensor) -> list[torch.Tensor]:
    """grid [..., n] (meshgrid of 1-D coords, last dim x, y[, z] over the reversed leading axes) -> the 1-D coords per
    axis (x first); asserts separability."""
    n = grid.shape[-1]
    sp = grid.shape[:-1]
    out = []
    for k in range(n):
        ax = len(sp) - 1 - k                                                   # x varies along the last axis
        idx = [0] * len(sp)
        idx[ax] = slice(None)
        c = grid[tuple(idx) + (k,)]
        shape = [1] * len(sp)
        shape[ax] = -1
        assert torch.equal(grid[..., k], c.view(shape).expand(sp)), f"grid not separable on axis {k}"
        out.append(c.float())
    return out


class DenseHeads(nn.Module):
    def __init__(self, occ_proj, occ_fuse, refiner, occ_pred, seg_decoder, g_o, mz_o, mz_v, g_m, z_ch, crop="grid"):
        super().__init__()
        self.occ_proj, self.occ_fuse, self.refiner, self.occ_pred, self.seg_decoder = occ_proj, occ_fuse, refiner, occ_pred, seg_decoder
        for k, v in dict(g_o=g_o, mz_o=mz_o, mz_v=mz_v, g_m=g_m).items():
            self.register_buffer(k, v)
        self.z_ch, self.crop = z_ch, crop

    def set_sizes(self, src: int):
        """Index / weight buffers of the lerp crop for an S x S source (both crops read the G x G BEV grid)."""
        for name, g in (("ox", self.g_o[0, 0, :, 0]), ("oy", self.g_o[0, :, 0, 1]),
                        ("mx", self.g_m[0, 0, :, 0]), ("my", self.g_m[0, :, 0, 1])):
            i, w = lerp_taps(g.cpu(), src)
            self.register_buffer(f"{name}_i", i.to(self.g_o.device))
            self.register_buffer(f"{name}_w", w.to(self.g_o.device))
        return self

    def lerp2(self, v, ax, ay):
        """[1, C, Ys, Xs] -> [1, C, Yt, Xt] by two gathers + weights per axis (x, then y)."""
        ix, wx = getattr(self, f"{ax}_i"), getattr(self, f"{ax}_w").to(v.dtype)
        v = v.index_select(3, ix[0]) * wx[0] + v.index_select(3, ix[1]) * wx[1]
        iy, wy = getattr(self, f"{ay}_i"), getattr(self, f"{ay}_w").to(v.dtype)
        return v.index_select(2, iy[0]) * wy[0][:, None] + v.index_select(2, iy[1]) * wy[1][:, None]

    def crop3(self, v, mz):
        """[1, C, Zs, Ys, Xs] -> [1, C, Zt, Yt, Xt]: x / y as a 4D grid_sample with Z in the channels (the occupancy grid
        is separable, so this is the 5D sample's x / y part exactly), then Z as a [Zt, Zs] matmul."""
        _, c, z, ys, xs = v.shape
        if self.crop == "lerp":
            v = self.lerp2(v.reshape(1, c * z, ys, xs), "ox", "oy")
        else:
            v = F.grid_sample(v.reshape(1, c * z, ys, xs), self.g_o.to(v.dtype), mode="bilinear", align_corners=True)
        yt, xt = v.shape[2:]
        # Z as a grouped 1x1 conv (channels c * Z, one group per c): a [Zt, Zs] x [C, Zs, Y X] batched GEMM (K = 16)
        # got a 5.9 ms TensorRT kernel
        wz = mz.to(v.dtype)[None].expand(c, -1, -1).reshape(c * mz.shape[0], z, 1, 1)
        return F.conv2d(v, wz, groups=c).reshape(1, c, mz.shape[0], yt, xt)

    def forward(self, bev, vol):
        _, c, h, w = bev.shape
        occ_b = self.crop3(bev.reshape(1, c // self.z_ch, self.z_ch, h, w), self.mz_o)
        occ_v = self.crop3(self.occ_proj(vol), self.mz_v)                     # zup folded into mz_v
        f = occ_b + self.occ_fuse(torch.cat([occ_b, occ_v], dim=1))
        occ = self.occ_pred(self.refiner(f).permute(0, 4, 3, 2, 1))
        if self.crop == "lerp":
            m = self.lerp2(bev, "mx", "my")
        else:
            m = F.grid_sample(bev, self.g_m.to(bev.dtype), mode="bilinear", align_corners=True)
        return occ, self.seg_decoder(m)

    @staticmethod
    def from_modules(defs: dict, z_ch: int, heads_ckpt: str | Path, refiner_ckpt: str | Path, grids: dict,
                     crop: str = "grid") -> "DenseHeads":
        """``defs``: NAMES -> the upstream modules (definitions, taken before the chain swaps them for engines);
        weights: the fp32 masters in heads_ckpt["heads"] where present, else the module's own."""
        from qdistill import zhalf
        from qdistill.occ2d import build as build_occ2d

        sd = torch.load(heads_ckpt, map_location="cpu", weights_only=False)["heads"]
        mods = {n: trt_modules.fp32_copy(defs[n], sd.get(n)) for n in NAMES}
        rk = torch.load(refiner_ckpt, map_location="cpu", weights_only=False)
        refiner = build_occ2d(rk["cfg"])
        refiner.load_state_dict(rk["model"])
        refiner = refiner.float().eval()
        g_o, (G, Zs) = grids["occ"], grids["occ_src"]                          # [1, Zt, Yt, Xt, 3] (x, y, z)
        _, _, gz = separable(g_o[0])                                           # asserts the 5D grid is separable
        mz = axis_matrix(gz, Zs)
        return DenseHeads(mods["uvtr_occ_proj"], mods["uvtr_occ_fuse"], refiner, mods["occ_pred_head"], mods["map_seg"],
                          g_o[:, 0, :, :, :2].float(), mz, mz @ zhalf.zup_matrix().float(), grids["map"].float()[None],
                          z_ch, crop).set_sizes(G).eval()


def capture_grids(head, metas) -> dict:
    """The occupancy / map sampling grids of the installed chain (warp_grid patches both), in the chain's dtype."""
    from qdistill import zhalf

    hd = head.bev_modeling.head
    tr = hd.transformer
    dt = next(hd.parameters()).dtype
    rng, vs = tr._get_occ_target_spec([metas])
    g, _, _ = tr._normalized_occ_grid(hd.bev_h, hd.bev_w, "cuda", dt, occ_pc_range=rng, occ_voxel_size=vs)
    x = torch.zeros(1, 1, hd.bev_h, hd.bev_w, device="cuda", dtype=dt)
    return {"occ": g.cpu(), "occ_src": (hd.bev_h, zhalf.Z16), "map": hd.feat_cropper._grid(x).type_as(x).cpu()}


class DenseEngine(nn.Module):
    """The engine: inputs bev / vol (cast to fp16), outputs occ / seg (copies, cast back to the input dtype)."""

    def __init__(self, path: str | Path):
        super().__init__()
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.path = str(path)
        self.shape = {n: tuple(self.engine.get_tensor_shape(n)) for n in ("bev", "vol", "occ", "seg")}
        self.out = {n: torch.empty(self.shape[n], dtype=torch.float16, device="cuda") for n in ("occ", "seg")}
        for n, t in self.out.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.scratch_MiB = round(self.engine.device_memory_size_v2 / 2**20, 1)

    def forward(self, bev, vol):
        ins = {"bev": bev.to(torch.float16).contiguous(), "vol": vol.to(torch.float16).contiguous()}
        for n, t in ins.items():
            assert tuple(t.shape) == self.shape[n], (self.path, n, tuple(t.shape), self.shape[n])
            self.ctx.set_input_shape(n, self.shape[n])
            self.ctx.set_tensor_address(n, t.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out["occ"].to(bev.dtype, copy=True), self.out["seg"].to(bev.dtype, copy=True)


class _Ret(nn.Module):
    """A head submodule stand-in returning what the engine produced for it (st[key])."""

    def __init__(self, st, key):
        super().__init__()
        self.st, self.key = st, key

    def forward(self, x):
        return self.st.pop(self.key)


class _Tiny(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return self.t


def install(head, engine: str | Path) -> dict:
    """After every other item (warp_grid, zhalf2 fold, mod_trt, heads_fp16): the transformer's occ path and the map
    head run as the engine. The upstream forward is kept; its occ / map steps become stand-ins: the BEV and volume
    adapters stash their inputs, the fuse runs the engine, occ_decoder / occ_pred_head / feat_cropper / seg_decoder
    return its outputs (all inside the same forward, so CUDA-graph capture sees one consistent sequence)."""
    hd = head.bev_modeling.head
    tr = hd.transformer
    assert getattr(tr, "_warp_occ", None) is not None, "dense_trt: built for the warped-grid chain (after warp_grid)"
    eng = DenseEngine(engine)
    st = {}
    tiny = torch.zeros(1, 1, 1, 1, 1, device="cuda", dtype=next(hd.parameters()).dtype)

    def adapt_bev(bev_feat, bev_h, bev_w, occ_pc_range=None, occ_voxel_size=None):
        st["bev"] = bev_feat
        return tiny

    def adapt_volume(volume_feat, bev_h, bev_w, occ_pc_range=None, occ_voxel_size=None):
        assert volume_feat.shape[2] == 8, f"dense_trt: built for the zhalf2 Z8 volume, got {tuple(volume_feat.shape)}"
        st["vol"] = volume_feat
        return tiny

    def fuse(bev_feat, uvtr_occ_feat):
        st["occ"], st["seg"] = eng(st.pop("bev"), st.pop("vol"))
        return tiny

    tr._adapt_bev_for_occ, tr._adapt_volume_for_occ, tr._fuse_uvtr_occ_feat = adapt_bev, adapt_volume, fuse
    tr.occ_decoder = _Tiny(tiny)
    tr.occ_pred_head = _Ret(st, "occ")
    hd.feat_cropper = _Tiny(tiny[0, 0])
    hd.seg_decoder = _Ret(st, "seg")
    head.bev_modeling._dense_engine = eng
    return {"dense_trt": str(engine), "scratch_MiB": eng.scratch_MiB, "shapes": eng.shape}



def self_test() -> dict:
    torch.manual_seed(0)
    G, Z = 15, 16
    # a warped-like monotone axis, partly outside [-1, 1] (zeros padding), and a plain one
    t = torch.linspace(-1.05, 1.05, 20)
    gx = torch.sign(t) * t.abs() ** 1.3
    gy = torch.linspace(-0.9, 0.95, 18)
    gz = torch.linspace(-1, 1, Z)
    zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing="ij")
    grid = torch.stack((xx, yy, zz), -1)[None]
    v = torch.randn(1, 3, Z, G, G)
    ref = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    cx, cy, cz = separable(grid[0])
    mx, my, mz = axis_matrix(cx, G), axis_matrix(cy, G), axis_matrix(cz, Z)
    out = torch.einsum("tz,bczyx->bctyx", mz, torch.einsum("sy,bczyx->bczsx", my, torch.einsum("rx,bczyx->bczyr", mx, v)))
    e3 = float((out - ref).abs().max())
    assert e3 < 1e-5, e3
    y2, x2 = torch.meshgrid(gy, gx, indexing="ij")
    g2 = torch.stack((x2, y2), -1)
    b = torch.randn(1, 4, G, G)
    r2 = F.grid_sample(b, g2[None], mode="bilinear", align_corners=True)
    c2x, c2y = separable(g2)
    o2 = axis_matrix(c2y, G) @ (b @ axis_matrix(c2x, G).t())
    e2 = float((o2 - r2).abs().max())
    assert e2 < 1e-5, e2
    for g, n in ((cx, G), (cy, G)):
        i, w = lerp_taps(g, n)
        m = torch.zeros(len(g), n)
        for k in range(2):
            m[torch.arange(len(g)), i[k]] += w[k]
        assert torch.allclose(m, axis_matrix(g, n), atol=1e-6)
    try:
        separable(grid[0] + torch.randn_like(grid[0]) * 1e-3)
        raise AssertionError("non-separable grid accepted")
    except AssertionError as e:
        assert "not separable" in str(e), e
    return {"crop3_max_abs": e3, "crop2_max_abs": e2, "non_separable": "rejected"}


if __name__ == "__main__":
    if sys.argv[1:] == ["self-test"]:
        print(self_test())
