"""Make the frozen teacher head differentiable w.r.t. its input taps.

Two upstream ops call a compiled kernel's ``forward`` only, so autograd stops (or raises) there; both compiled
modules also bind a CUDA backward that upstream never wires in. ``enable_head_grad()`` patches both:
  * ``attention.multi_scale_deformable_attn_cuda``
    -> a dispatch that wraps forward + backward in ``MSDeformAttnBF16``;
  * ``ops._VoxelPoolDepthCuda`` (lift step: image features x depth distribution summed into voxels, called by
    ``ops.voxel_pool_depth`` from ``view_transform``) -> ``VoxelPoolDepthGrad``, which adds
    ``voxel_pool_depth_backward_all`` (grad_feat = sum over the point's voxel of grad_out x depth, atomics;
    grad_depth = <grad_out, feat> over channels). Without it, backward through the head raises
    NotImplementedError.
``third_party`` code is not edited.
"""

from __future__ import annotations

import torch

from qwen_drive_perception import attention, ops
from qwen_drive_perception.layers import multi_scale_deformable_attn_pytorch
from qwen_drive_perception.ops import _load_ms_deform_attn_bf16, _load_voxel_pool

_ORIGINAL = attention.multi_scale_deformable_attn_cuda
_ORIGINAL_VOXEL_POOL = ops._VoxelPoolDepthCuda


class MSDeformAttnBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        ext = _load_ms_deform_attn_bf16()
        ctx.im2col_step = int(im2col_step)
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return ext.forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, ctx.im2col_step)

    @staticmethod
    def backward(ctx, grad_output):
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        ext = _load_ms_deform_attn_bf16()
        grad_value, grad_loc, grad_attn = ext.backward(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights,
            grad_output.contiguous(), ctx.im2col_step,
        )
        return grad_value, None, None, grad_loc, grad_attn, None


def _differentiable_dispatch(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                             attention_weights, im2col_step):
    if not value.is_cuda:
        return multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
    if value.dtype == torch.float32:  # exact (fp32) encoder in the loop: PyTorch fallback, autograd via grid_sample
        return multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations.float(),
                                                   attention_weights.float())
    if value.dtype != torch.bfloat16:
        raise TypeError(f"the ms_deform_attn kernel requires CUDA bfloat16 tensors, got {value.dtype}")
    return MSDeformAttnBF16.apply(
        value.contiguous(), value_spatial_shapes.contiguous(), value_level_start_index.contiguous(),
        sampling_locations.contiguous(), attention_weights.contiguous(), int(im2col_step),
    )


class VoxelPoolDepthGrad(torch.autograd.Function):
    """Upstream forward unchanged; adds the compiled all-camera backward.

    Output [B, N_cam, X, Y, Z, C] (fp32 kernel output cast to promote(feat, depth)). Each valid point (image, d, h, w)
    lands in exactly one voxel, so grad_depth is a plain write and grad_feat an atomic sum over depth bins.
    """

    @staticmethod
    def forward(ctx, img_feats, img_depth, coords, point_indices, ranks, B, N_sweep, N_cam, X, Y, Z, D, H, W):
        out = _ORIGINAL_VOXEL_POOL.forward(ctx, img_feats, img_depth, coords, point_indices, ranks,
                                           B, N_sweep, N_cam, X, Y, Z, D, H, W)
        ctx.save_for_backward(img_feats, img_depth, coords.int().contiguous(), point_indices.int().contiguous())
        ctx.dims = (B, N_sweep, N_cam, X, Y, Z, D, H, W)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        img_feats, img_depth, coords, point_indices = ctx.saved_tensors
        grad_feats, grad_depth = _load_voxel_pool().voxel_pool_depth_backward_all(
            grad_out.contiguous(), img_feats, img_depth, coords, point_indices, *ctx.dims)
        return (grad_feats.to(img_feats.dtype) if ctx.needs_input_grad[0] else None,
                grad_depth.to(img_depth.dtype) if ctx.needs_input_grad[1] else None,
                *([None] * 12))


def enable_head_grad(enable: bool = True) -> None:
    attention.multi_scale_deformable_attn_cuda = _differentiable_dispatch if enable else _ORIGINAL
    ops._VoxelPoolDepthCuda = VoxelPoolDepthGrad if enable else _ORIGINAL_VOXEL_POOL
