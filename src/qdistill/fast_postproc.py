"""A: deployment read-out (boxes, occupancy labels, map labels to the CPU) with one host sync instead of one per
tensor.

    from qdistill import fast_postproc
    post = fast_postproc.FastPost()
    out = post(head, outs, metas)     # same dict as e2e_latency.deploy_outputs: {"det": [...], "occ": ..., "map": ...}

Upstream (heads.get_bboxes / get_occ / get_map_seg) syncs the host five or more times per frame: torch.linalg.inv
checks errors on the host, the post-center mask indexes with a data-dependent count, and each of boxes / scores / labels
/ occ / map is copied with a blocking .cpu(). The GPU idles through each. Here the
same maths runs on the GPU without syncs (torch.linalg.inv_ex without the error check, the mask kept as a tensor),
everything is copied non-blocking into pinned buffers, the host waits once, and the mask is applied on the CPU.
Same ops, dtypes and order as upstream, so the outputs are bit-identical (e2e_latency --fast-post checks it).
"""

from __future__ import annotations

import torch

from qwen_drive_perception import geometry
from qwen_drive_perception.heads import denormalize_bbox


class FastPost:
    def __init__(self):
        self.pinned: dict = {}

    def _host(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """Non-blocking copy into a reused pinned buffer of the same shape / dtype."""
        b = self.pinned.get(name)
        if b is None or b.shape != t.shape or b.dtype != t.dtype:
            b = self.pinned[name] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        b.copy_(t, non_blocking=True)
        return b

    @torch.no_grad()
    def __call__(self, head, outs: dict, metas: dict) -> dict:
        coder = head.bbox_coder
        cls_scores = outs["all_cls_scores"][-1][0].sigmoid()                         # NMSFreeCoder.decode_single
        scores, indexs = cls_scores.view(-1).topk(coder.max_num)
        labels = indexs % coder.num_classes
        bbox_preds = outs["all_bbox_preds"][-1][0][indexs // coder.num_classes]
        boxes = denormalize_bbox(bbox_preds, coder.pc_range)
        pcr = torch.tensor(coder.post_center_range, device=scores.device)
        mask = (boxes[..., :3] >= pcr[:3]).all(1) & (boxes[..., :3] <= pcr[3:]).all(1)
        if metas.get("box_coord_system") == "ego":                                    # get_bboxes
            l2e = metas["lidar2ego"]
            l2e = l2e[0] if isinstance(l2e, torch.Tensor) else torch.as_tensor(l2e[0], dtype=torch.float32)
            ego2lidar = torch.linalg.inv_ex(l2e.to(boxes.device).to(dtype=torch.float32))[0]  # = linalg.inv, no host check
            boxes = geometry._transform_boxes(boxes, ego2lidar.to(boxes.dtype))
        boxes = boxes.clone()
        boxes[:, 2] = boxes[:, 2] - boxes[:, 5] * 0.5
        occ = head.get_occ(outs, [metas])[0].byte()
        mp = head.get_map_seg(outs, [metas])[0].byte()
        h = {n: self._host(n, t) for n, t in (("boxes", boxes.float()), ("scores", scores.float()), ("labels", labels),
                                               ("mask", mask), ("occ", occ), ("map", mp))}
        torch.cuda.current_stream().synchronize()
        m = h["mask"].numpy()
        det = {"boxes": h["boxes"].numpy()[m], "scores": h["scores"].numpy()[m], "labels": h["labels"].numpy()[m]}  # get_bboxes(...)[0]
        return {"det": det, "occ": h["occ"].numpy().copy(), "map": h["map"].numpy().copy()}
