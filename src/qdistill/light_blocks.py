"""Stage-2 light replacements for the frozen head's dense 3D blocks.

LightVoxelEncoder replaces Uni3DVoxelPoolDepth.feat_encoding's three 256-channel 3x3x3 Conv3d (224 ms) and the 68 ms
permute copy before them. It reads the camera-summed pooling output in the kernel's native [B, 1, X, Y, Z, C] layout
as a channels_last_3d [B, C, X, Y, Z] view (no copy), works at ``mid`` channels, permutes only the ``mid``-channel
tensor to [B, mid, Z, Y, X] and expands back to 256 there, so the output has upstream's layout and meaning
([B, 256, Z, Y, X], post-ReLU) for the frozen consumers (uvtr_query_proj, uvtr_occ_proj).

TeacherShapedVoxelEncoder is upstream's own stack, optionally rank-factored and initialised from the teacher.

install_light_voxel() swaps either into a loaded head (instance-level patch of view_trans.forward; upstream untouched).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch import nn


def _cbr(cin, cout, k, dilation=1):
    return nn.Sequential(nn.Conv3d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
                         nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class LightVoxelEncoder(nn.Module):
    """1x1 256->mid, ``n_mid`` x (3x3x3 mid->mid, residual), [permute at mid channels], 1x1 mid->256 + BN + ReLU.

    ``dilations`` (one per mid conv, default all 1) sets the receptive field without changing depth or parameters:
    1 + 2 * sum(dilations) voxels ((1, 1) -> 5, (1, 2) -> 7, upstream's 3 convs -> 7)."""

    def __init__(self, channels: int = 256, mid: int = 64, n_mid: int = 2, dilations: tuple[int, ...] | None = None):
        super().__init__()
        dilations = tuple(dilations or (1,) * n_mid)
        assert len(dilations) == n_mid, (dilations, n_mid)
        self.cfg = {"channels": channels, "mid": mid, "n_mid": n_mid, "dilations": dilations}
        self.reduce = _cbr(channels, mid, 1)
        self.mid = nn.ModuleList(_cbr(mid, mid, 3, d) for d in dilations)
        self.expand = _cbr(mid, channels, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """``s``: [B, 1, X, Y, Z, C] contiguous (camera sum of voxel_pool_depth). Returns [B, C, Z, Y, X] contiguous."""
        x = s[:, 0].permute(0, 4, 1, 2, 3)  # [B, C, X, Y, Z], channels_last_3d strides, no copy
        x = self.reduce(x)
        for blk in self.mid:
            x = x + blk(x)
        x = x.permute(0, 1, 4, 3, 2).contiguous()  # [B, mid, Z, Y, X]: the only layout copy, at mid channels
        return self.expand(x)


class TeacherShapedVoxelEncoder(nn.Module):
    """Upstream's stack, 3 x (3x3x3 Conv3d 256->256 + BN + ReLU), in the kernel's native [B, C, X, Y, Z] layout, with
    each conv optionally factored to rank r: 3x3x3 256->r, then 1x1 r->256. Depth, kernel, receptive field (7),
    nonlinearities and output match upstream; only the rank differs. ``bn=False`` for a teacher init (BN folded into
    the convs, init_from_teacher), True for a random init. The single layout copy to [B, C, Z, Y, X] happens inside the
    last layer, at r channels when factored."""

    def __init__(self, channels: int = 256, rank: int | list[int] | None = None, n_layers: int = 3, bn: bool = False,
                 checkpoint: bool = False, kind: str = "teacher_shaped"):
        """``rank``: None (full), one int for every layer, or one per layer ([32, 64, 96])."""
        super().__init__()
        if isinstance(rank, (list, tuple)):
            rank = [int(r) for r in rank]
            assert len(rank) == n_layers, (rank, n_layers)
        self.cfg = {"kind": kind, "channels": channels, "rank": rank, "n_layers": n_layers, "bn": bn}
        self.checkpoint = checkpoint
        self.ranks = list(rank) if isinstance(rank, list) else [rank] * n_layers
        ks = [r or channels for r in self.ranks]
        self.spatial = nn.ModuleList(nn.Conv3d(channels, k, 3, padding=1, bias=rank is None) for k in ks)
        self.proj = nn.ModuleList(nn.Conv3d(k, channels, 1, bias=True) for k in ks) if rank else None
        self.bn = nn.ModuleList(nn.BatchNorm3d(channels) for _ in range(n_layers)) if bn else None

    def _layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        y = self.spatial[i](x)
        if i == len(self.spatial) - 1:
            y = y.permute(0, 1, 4, 3, 2).contiguous()  # [B, k, X, Y, Z] -> [B, k, Z, Y, X]
        if self.proj is not None:
            y = self.proj[i](y)
        if self.bn is not None:
            y = self.bn[i](y)
        return torch.relu(y)

    def forward(self, s: torch.Tensor, return_layers: bool = False):
        """``s``: [B, 1, X, Y, Z, C] contiguous (camera sum of voxel_pool_depth). Returns [B, C, Z, Y, X] contiguous;
        with ``return_layers`` the list of every layer's output."""
        x = s[:, 0].permute(0, 4, 1, 2, 3)
        outs = []
        for i in range(len(self.spatial)):
            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(self._layer, i, x, use_reentrant=False)
            else:
                x = self._layer(i, x)
            outs.append(x)
        return outs if return_layers else x


class FactorisedVoxelEncoder(nn.Module):
    """TeacherShapedVoxelEncoder with each 3x3x3 conv split into two cheaper ones (layout [B, C, X, Y, Z], kernel
    dims (kx, ky, kz)): ``split="xy_z"`` 3x3x1 C->r then 1x1x3 r->r (12 taps; receptive field per layer 3 x 3 x 3, as
    upstream), ``split="asym"`` Cylinder3D-style 3x1x3 C->r then 1x3x3 r->r (18 taps; 3 x 3 x 5). Then 1x1 r->C + ReLU
    as the rank-factored block (ReLU also between the two). Same interface / output layout as TeacherShapedVoxelEncoder."""

    SPLITS = {"xy_z": ((3, 3, 1), (1, 1, 3)), "asym": ((3, 1, 3), (1, 3, 3))}

    def __init__(self, channels: int = 256, rank: int = 64, n_layers: int = 3, split: str = "xy_z",
                 checkpoint: bool = False, kind: str = "factorised"):
        super().__init__()
        self.cfg = {"kind": kind, "channels": channels, "rank": rank, "n_layers": n_layers, "split": split}
        self.checkpoint = checkpoint
        ka, kb = self.SPLITS[split]
        pad = lambda k: tuple(x // 2 for x in k)
        self.a = nn.ModuleList(nn.Conv3d(channels, rank, ka, padding=pad(ka)) for _ in range(n_layers))
        self.b = nn.ModuleList(nn.Conv3d(rank, rank, kb, padding=pad(kb)) for _ in range(n_layers))
        self.proj = nn.ModuleList(nn.Conv3d(rank, channels, 1) for _ in range(n_layers))

    def _layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        y = self.b[i](torch.relu(self.a[i](x)))
        if i == len(self.a) - 1:
            y = y.permute(0, 1, 4, 3, 2).contiguous()  # [B, r, X, Y, Z] -> [B, r, Z, Y, X], at r channels
        return torch.relu(self.proj[i](y))

    def forward(self, s: torch.Tensor, return_layers: bool = False):
        x = s[:, 0].permute(0, 4, 1, 2, 3)
        outs = []
        for i in range(len(self.a)):
            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(self._layer, i, x, use_reentrant=False)
            else:
                x = self._layer(i, x)
            outs.append(x)
        return outs if return_layers else x


class TriPlanarVoxelEncoder(nn.Module):
    """, two tri-plane variants of the block (same interface / output layout as TeacherShapedVoxelEncoder; layout
    [B, C, X, Y, Z]). Per layer, then 1x1 r->C + ReLU:
    ``mode="tri"``: three parallel planar convs C->r, 3x3x1 (XY) + 3x1x3 (XZ) + 1x3x3 (YZ), summed + ReLU (27 taps in
      all, as one 3x3x3, but three planar kernels; receptive field per layer the 3 x 3 x 3 cross, not the cube).
    ``mode="triplane"``: TPV / EG3D-style planes: the volume mean-pooled over Z, Y and X into XY, XZ, YZ planes, a 3x3
      Conv2d C->r on each, broadcast back and summed with a per-voxel 1x1x1 C->r (keeps the voxel's own features) +
      ReLU. No 3D spatial conv: plane context is global along the pooled axis."""

    def __init__(self, channels: int = 256, rank: int = 32, n_layers: int = 3, mode: str = "triplane",
                 checkpoint: bool = False, kind: str = "triplanar"):
        super().__init__()
        assert mode in ("tri", "triplane"), mode
        self.cfg = {"kind": kind, "channels": channels, "rank": rank, "n_layers": n_layers, "mode": mode}
        self.mode, self.checkpoint = mode, checkpoint
        if mode == "tri":
            ks = ((3, 3, 1), (3, 1, 3), (1, 3, 3))
            self.planes = nn.ModuleList(nn.ModuleList(nn.Conv3d(channels, rank, k, padding=tuple(x // 2 for x in k))
                                                      for k in ks) for _ in range(n_layers))
        else:
            self.planes = nn.ModuleList(nn.ModuleList(nn.Conv2d(channels, rank, 3, padding=1) for _ in range(3))
                                        for _ in range(n_layers))
            self.local = nn.ModuleList(nn.Conv3d(channels, rank, 1) for _ in range(n_layers))
        self.proj = nn.ModuleList(nn.Conv3d(rank, channels, 1) for _ in range(n_layers))

    def _layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p = self.planes[i]
        if self.mode == "tri":
            y = p[0](x) + p[1](x) + p[2](x)
        else:
            xy, xz, yz = p[0](x.mean(4)), p[1](x.mean(3)), p[2](x.mean(2))       # [B, r, X, Y] / [X, Z] / [Y, Z]
            y = self.local[i](x) + xy[..., None] + xz[:, :, :, None, :] + yz[:, :, None]
        y = torch.relu(y)
        if i == len(self.proj) - 1:
            y = y.permute(0, 1, 4, 3, 2).contiguous()  # [B, r, X, Y, Z] -> [B, r, Z, Y, X], at r channels
        return torch.relu(self.proj[i](y))

    def forward(self, s: torch.Tensor, return_layers: bool = False):
        x = s[:, 0].permute(0, 4, 1, 2, 3)
        outs = []
        for i in range(len(self.proj)):
            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(self._layer, i, x, use_reentrant=False)
            else:
                x = self._layer(i, x)
            outs.append(x)
        return outs if return_layers else x


def _folded(conv, bn):
    """Upstream conv + eval-mode BN as one affine conv, kernel reordered (kz, ky, kx) -> (kx, ky, kz) for [B,C,X,Y,Z]."""
    scale = bn.weight.float() / (bn.running_var.float() + bn.eps).sqrt()
    w = conv.weight.float() * scale[:, None, None, None, None]
    b = (conv.bias.float() - bn.running_mean.float()) * scale + bn.bias.float()
    return w.permute(0, 1, 4, 3, 2), b


@torch.no_grad()
def teacher_preact_stats(vt, volumes) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Mean and covariance (fp64, over voxels) of each upstream layer's post-BN, pre-ReLU output, for the pooled
    volumes ``volumes`` (iterable of [B, 1, X, Y, Z, C]). Used by init_from_teacher(basis="pca")."""
    C = vt.conv_layer[0][0].out_channels
    acc = None
    n = 0
    for s in volumes:
        if acc is None:
            acc = [[torch.zeros(C, dtype=torch.float64, device=s.device),
                    torch.zeros(C, C, dtype=torch.float64, device=s.device)] for _ in vt.conv_layer]
        x = s.permute(0, 1, 5, 4, 3, 2).sum(1)
        del s
        for a, (conv, bn, _) in zip(acc, vt.conv_layer):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = bn(conv(x))
            r = y.movedim(1, -1).reshape(-1, C)
            for c0 in range(0, r.shape[0], 65536):
                c = r[c0:c0 + 65536].float()
                a[0] += c.sum(0).double()
                a[1] += (c.T @ c).double()
            nv = r.shape[0]
            del r
            x = torch.relu_(y)
        n += nv
        del x, y
    out = []
    for sx, sxx in acc:
        m = sx / n
        out.append((m, sxx / n - torch.outer(m, m)))
    return out


@torch.no_grad()
def init_from_teacher(block: TeacherShapedVoxelEncoder, vt, basis: str = "svd", stats=None,
                      teacher_layers: tuple[int, ...] | None = None) -> None:
    """Copy upstream's convs (BN folded) into ``block``. Factored layers keep the best rank-r part of each conv:
    "svd" of the folded weight [256, 256*27] (weight space), or "pca": project the layer's output onto its top-r
    principal directions on real volumes (``stats`` from teacher_preact_stats; output space, data-aware).
    ``teacher_layers``: which upstream layer initialises each block layer (default all, in order); a block with fewer
    layers than upstream skips some ((0, 2) drops the middle one; layer 2 then sees layer 0's output)."""
    assert not block.cfg["bn"], "teacher init folds BN; build the block with bn=False"
    teacher_layers = tuple(teacher_layers or range(len(vt.conv_layer)))
    assert len(teacher_layers) == len(block.spatial), (teacher_layers, len(block.spatial))
    for i, j in enumerate(teacher_layers):
        conv, bn, _ = vt.conv_layer[j]
        w, b = _folded(conv, bn)
        r = block.ranks[i]
        if r is None:
            block.spatial[i].weight.copy_(w)
            block.spatial[i].bias.copy_(b)
            continue
        m = w.reshape(w.shape[0], -1)
        if basis == "svd":
            u, sv, vh = torch.linalg.svd(m, full_matrices=False)
            left, right, bias = u[:, :r] * sv[:r].sqrt(), sv[:r].sqrt()[:, None] * vh[:r], b
        elif basis == "pca":
            mean, cov = (t.float() for t in stats[j])
            v = torch.linalg.eigh(cov)[1].flip(1)[:, :r]
            left, right, bias = v, v.T @ m, v @ (v.T @ (b - mean)) + mean
        else:
            raise ValueError(basis)
        block.spatial[i].weight.copy_(right.reshape(r, *w.shape[1:]))
        block.proj[i].weight.copy_(left[:, :, None, None, None])
        block.proj[i].bias.copy_(bias)






def pooled_volume(vt, mlvl_feats, img_depth, img_metas) -> torch.Tensor:
    """Upstream view_trans up to the camera sum, in the kernel's layout: [B, 1, X, Y, Z, C]."""
    from qwen_drive_perception.ops import voxel_pool_depth

    for m in img_metas:
        m["lidar2img"] = vt._format_lidar2img(m["lidar2img"][None, ...])
    voxel_coords, mask = vt.coord_preparing(img_metas)
    img_feats, depth = mlvl_feats[0], img_depth[0]
    B, X, Y, Z = len(img_feats), *vt.voxel_shape
    v = voxel_pool_depth(img_feats, depth, voxel_coords, mask, B, X, Y, Z)
    return v.sum(1, keepdim=True)


def teacher_encoding(vt, s: torch.Tensor) -> torch.Tensor:
    """The frozen upstream feat_encoding on a pooled volume (the distillation target): [B, C, Z, Y, X]."""
    return vt.feat_encoding(s.permute(0, 1, 5, 4, 3, 2).contiguous())


def build_voxel_block(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    mode = cfg.pop("z8_compress", None)
    if mode:  # 16 -> 8 compression inside the block (zhalf v2); the stride modes rebuild the first conv
        from qdistill.zhalf import CompressedZ8Block
        return CompressedZ8Block(build_voxel_block(cfg), mode, cfg.get("channels", 256))
    if cfg.pop("zhalf", False):  # the block at 8 height levels inside the 16-level interface
        from qdistill.zhalf import ZHalfBlock
        return ZHalfBlock(build_voxel_block(cfg))
    kind = cfg.pop("kind", "light")
    if kind == "teacher_shaped":
        return TeacherShapedVoxelEncoder(**cfg)
    if kind == "factorised":
        return FactorisedVoxelEncoder(**cfg)
    if kind == "triplanar":
        return TriPlanarVoxelEncoder(**cfg)
    return LightVoxelEncoder(**cfg)


def load_light_voxel(path: str | Path, device="cuda") -> nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    block = build_voxel_block(ck["cfg"])
    block.load_state_dict(ck["model"])
    return block.to(device).eval()


@contextmanager
def install_light_voxel(head, block: nn.Module):
    """Run the loaded head with ``block`` in place of the voxel-pool conv stack (and its permute)."""
    vt = head.bev_modeling.view_trans

    def forward(mlvl_feats, img_depth, img_metas):
        s = pooled_volume(vt, mlvl_feats, img_depth, img_metas)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return block(s)

    vt.forward = forward
    try:
        yield
    finally:
        del vt.forward
