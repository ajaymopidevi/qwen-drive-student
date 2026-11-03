"""voxel pooling with the cameras merged in the kernel, written straight as the voxel engine's fp16
input [B, 1, X, Y, Z, C].

Upstream pooled_volume (light_blocks): voxel_pool_depth -> fp32 [B, N_cam, X, Y, Z, C] (3.93 GB, zero-filled) ->
.to(bf16) -> sum over cameras -> (_Engine) .to(fp16): 72.8 ms and 5.55 GiB in the deployed chain, of
which ~4 ms is the actual reduce. ``merged_pooled_volume`` keeps upstream's geometry (vt.coord_preparing, per frame)
and point order, and by default reproduces its rounding (``faithful=True``, see csrc/merged_pool_cuda.cu).

Point order: upstream sorts points by rank = camera-major (b, cam, x, y, z) with ``ranks.argsort()`` and reduces each
(camera, voxel) interval in that order. Here the same argsort is taken first, then a stable sort by voxel: intervals
become one per voxel, cameras inside in ascending order, and each camera's points keep upstream's order.

``install(head, trt_head)`` replaces view_trans.forward of a TrtHead-installed head (voxel engine) with
coords -> merged pooling -> engine.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None
_FLAGS = ["-O3", "--use_fast_math", "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
          "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"]


def _ext():
    global _EXT
    if _EXT is None:
        d = Path(__file__).parent / "csrc"
        _EXT = load(name="qdistill_merged_pool", sources=[str(d / "merged_pool.cpp"), str(d / "merged_pool_cuda.cu")],
                    extra_cuda_cflags=_FLAGS, verbose=os.getenv("QDISTILL_OPS_VERBOSE", "0") == "1")
    return _EXT


def merged_prep(voxel_coords, mask, B, X, Y, Z):
    """voxel_coords / mask [B, N_sweep, N_cam, D, H, W, (4)] -> (sorted point indices, interval starts / lengths /
    voxel ids, dims), all int32."""
    _, N_sweep, N_cam, D, H, W = mask.shape
    flat_mask = mask.reshape(-1)
    point_indices = torch.arange(flat_mask.numel(), device=mask.device, dtype=torch.int32)[flat_mask]
    coords = voxel_coords.reshape(-1, 4)[flat_mask].int()
    camera = (point_indices // (D * H * W)) % N_cam
    ranks = coords[:, 0] * N_cam * X * Y * Z + camera * X * Y * Z + coords[:, 1] * Y * Z + coords[:, 2] * Z + coords[:, 3]
    order = ranks.argsort()  # upstream's order (camera-major)
    vox = ((coords[order, 0] * X + coords[order, 1]) * Y + coords[order, 2]) * Z + coords[order, 3]
    vox_sorted, o2 = torch.sort(vox, stable=True)  # voxel-major; cameras ascending, upstream order within a camera
    order = order[o2]
    sorted_points = point_indices[order].contiguous()
    if vox_sorted.numel() == 0:
        e = torch.empty((0,), device=mask.device, dtype=torch.int32)
        return sorted_points, e, e, e, (N_sweep, N_cam, D, H, W)
    kept = torch.ones_like(vox_sorted, dtype=torch.bool)
    kept[1:] = vox_sorted[1:] != vox_sorted[:-1]
    starts = torch.where(kept)[0].int()
    lengths = torch.empty_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = vox_sorted.numel() - starts[-1]
    return sorted_points, starts, lengths, vox_sorted[starts.long()].int().contiguous(), (N_sweep, N_cam, D, H, W)


def merged_pool(img_feats, img_depth, prep, B, X, Y, Z, faithful=True, out=None, scale=None, dense=False):
    """-> fp16 [B, 1, X, Y, Z, C] (the voxel engine's input), cameras summed. ``out``: a static buffer to pool into
    (graph_head: the captured graph's input, no copy). , both bit-exact: ``scale`` fp16 [X, Y], the volume times
    it (in the kernel, = the eager s * scale); ``dense``: every voxel written by the kernel, no zero fill."""
    sorted_points, starts, lengths, voxels, (N_sweep, N_cam, D, H, W) = prep
    C = img_feats.shape[2]
    if out is None:
        out = (torch.empty if dense else torch.zeros)((B, 1, X, Y, Z, C), dtype=torch.float16, device=img_feats.device)
    else:
        assert out.shape == (B, 1, X, Y, Z, C) and out.dtype == torch.float16 and out.is_contiguous(), out.shape
        if not dense:
            out.zero_()
    none = torch.empty(0, dtype=torch.int32, device=img_feats.device)
    vi = none
    if dense:
        vi = torch.full((B * X * Y * Z,), -1, dtype=torch.int32, device=img_feats.device)
        vi[voxels.long()] = torch.arange(len(voxels), dtype=torch.int32, device=img_feats.device)
    sc = none.half() if scale is None else scale.to(torch.float16).contiguous().reshape(-1)
    _ext().merged_pool(img_feats.contiguous(), img_depth.contiguous(), sorted_points, starts, lengths, voxels, out,
                       N_sweep, N_cam, D, H, W, int(faithful), sc, vi, Z, X * Y)
    return out


def merged_pooled_volume(vt, mlvl_feats, img_depth, img_metas, faithful=True, out=None):
    """Drop-in for light_blocks.pooled_volume (+ the engine's fp16 cast): same geometry, fp16 [B, 1, X, Y, Z, C]."""
    for m in img_metas:
        m["lidar2img"] = vt._format_lidar2img(m["lidar2img"][None, ...])
    voxel_coords, mask = vt.coord_preparing(img_metas)
    img_feats, depth = mlvl_feats[0], img_depth[0]
    B, X, Y, Z = len(img_feats), *vt.voxel_shape
    return merged_pool(img_feats, depth, merged_prep(voxel_coords, mask, B, X, Y, Z), B, X, Y, Z, faithful, out)


def install(head, trt_head, faithful=True):
    """view_trans.forward -> coords, merged pooling, voxel engine (needs TrtHead with a voxel engine)."""
    vt = head.bev_modeling.view_trans
    assert hasattr(trt_head, "voxel"), "merged pooling feeds the TensorRT voxel engine (TrtHead voxel_engine)"

    # graph_head splits view_trans at the pooled volume: _pool (data-dependent shapes, eager) and _block (static)
    vt._pool = lambda mlvl_feats, img_depth, img_metas, out=None: merged_pooled_volume(
        vt, mlvl_feats, img_depth, img_metas, faithful, out)
    vt._block = lambda s: trt_head.voxel(s)                     # a plain callable: a Module would register as a child

    def forward(mlvl_feats, img_depth, img_metas):
        return vt._block(vt._pool(mlvl_feats, img_depth, img_metas))

    vt.forward = forward
