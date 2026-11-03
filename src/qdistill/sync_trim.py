"""A: fewer host syncs in voxel pooling, same maths.

    from qdistill import sync_trim
    sync_trim.install(head)        # after merged_pool / fast_geom installs

: the GPU idles ~2.6 ms per frame inside voxel pooling while the host waits on
  * torch.inverse(lidar2img): linalg_inv_ex + a host-side error check (sync); here linalg.inv_ex without the check
    (same kernel, same values);
  * merged_pool.merged_prep: two boolean-mask index ops on the same mask (one nonzero + sync each); here one nonzero
    and index_select (same values, same order).
The remaining sync (torch.where on the voxel-run starts) sizes the interval arrays and stays.
"""

from __future__ import annotations

import torch

from qdistill import fast_geom, merged_pool

_orig_prep = merged_pool.merged_prep


def merged_prep(voxel_coords, mask, B, X, Y, Z):
    """merged_pool.merged_prep with one nonzero for both index ops."""
    _, N_sweep, N_cam, D, H, W = mask.shape
    idx = mask.reshape(-1).nonzero().squeeze(1)
    point_indices = idx.int()
    coords = voxel_coords.reshape(-1, 4).index_select(0, idx).int()
    camera = (point_indices // (D * H * W)) % N_cam
    ranks = coords[:, 0] * N_cam * X * Y * Z + camera * X * Y * Z + coords[:, 1] * Y * Z + coords[:, 2] * Z + coords[:, 3]
    order = ranks.argsort()
    vox = ((coords[order, 0] * X + coords[order, 1]) * Y + coords[order, 2]) * Z + coords[order, 3]
    vox_sorted, o2 = torch.sort(vox, stable=True)
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


def _inverse_nocheck(x, *, out=None):
    assert out is None
    return torch.linalg.inv_ex(x)[0]


def install(head, mode: str = "einsum") -> dict:
    vt = head.bev_modeling.view_trans
    merged_pool.merged_prep = merged_prep                         # merged_pooled_volume looks it up at call time

    def coord_preparing(img_metas):
        inv = torch.inverse
        torch.inverse = _inverse_nocheck                          # fast_geom calls torch.inverse once per frame
        try:
            return fast_geom.coord_preparing(vt, img_metas, mode)
        finally:
            torch.inverse = inv
    vt.coord_preparing = coord_preparing
    return {"sync_trim": mode}
