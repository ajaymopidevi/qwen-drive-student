"""Head-runtime step 3 (voxel pooling glue 19 ms, of which ~8.8 ms is cuBLAS gemvx): view_trans.coord_preparing
with the two point transforms as GEMMs.

Upstream unprojects each of the 6 x 211,456 frustum points with a batched 4x4 @ 4x1 matmul (inverse lidar2img, then
lidar2ego): cuBLAS runs ~1.3 M tiny gemvs. Here the same products run as one [P, 4] x [4, 4] GEMM per camera
(``einsum``). No per-frame cache is possible: lidar2img folds in the ego motion to each camera's trigger, so every
frame's matrices differ (1,201 distinct of 1,201).

Not guaranteed bit-identical: the GEMM may sum the four products in another order, and ``.int()`` truncation turns a
last-bit difference at a voxel boundary into another voxel.

    fast_geom.install(head)    # view_trans.coord_preparing -> coord_preparing(view_trans, img_metas)
"""

from __future__ import annotations

import torch


def coord_preparing(vt, img_metas, mode: str = "einsum"):
    """Same outputs as Uni3DVoxelPoolDepth.coord_preparing: voxel_coords [B, N, C, D, H, W, 4] int32, mask bool.
    ``mode``: "einsum" (two GEMMs, upstream's operation order) or "fused" (lidar2ego @ inv(lidar2img) first, one GEMM)."""
    B = len(img_metas)
    frustum = vt.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    W, H, D = frustum.shape[1:-1]
    lidar2img = vt._stack_lidar2img(img_metas, frustum)
    _, N, C = lidar2img.shape[:3]
    lidar2img = lidar2img.reshape(B, -1, 4, 4)                               # [B, cameras x sweeps, 4, 4]
    pts = torch.cat([frustum, torch.ones_like(frustum[..., :1])], -1).flatten(1, 3)  # [B, P, 4]
    pts[..., :2] *= pts[..., 2:3]
    lidar2ego = vt._extract_lidar2ego(img_metas, frustum)                   # [B, 4, 4]
    inv = torch.inverse(lidar2img)
    if mode == "fused":
        pts = torch.einsum("bnij,bpj->bnpi", lidar2ego[:, None] @ inv, pts)
    else:
        pts = torch.einsum("bnij,bpj->bnpi", inv, pts)
        pts = torch.einsum("bij,bnpj->bnpi", lidar2ego, pts)
    pc_range = pts.new_tensor(vt.pc_range)
    voxel_size = pts.new_tensor(vt.voxel_size)
    voxel_coords = ((pts[..., :3] - pc_range[:3]) / voxel_size).int()
    batch_ix = torch.cat([torch.full_like(voxel_coords[ix : ix + 1, ..., 0:1], ix) for ix in range(B)], dim=0)
    voxel_coords = torch.cat([batch_ix, voxel_coords], dim=-1)
    voxel_coords = voxel_coords.view(B, N, C, W, H, D, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
    X, Y, Z = vt.voxel_shape
    mask = ((voxel_coords[..., 1] >= 0) & (voxel_coords[..., 1] < X) & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < Y) & (voxel_coords[..., 3] >= 0) & (voxel_coords[..., 3] < Z))
    return voxel_coords, mask


def install(head, mode: str = "einsum"):
    vt = head.bev_modeling.view_trans
    vt.coord_preparing = lambda img_metas: coord_preparing(vt, img_metas, mode)
    return {"fast_geom": mode}
