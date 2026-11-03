"""A: the head after voxel pooling replayed as one CUDA graph.

    from qdistill import graph_head
    graph_decoder.install(head)            # first (graph_head runs the graphed decoder's body inline)
    gh = graph_head.install(head)          # last, after every head_chain item (inference only)

Voxel pooling has data-dependent shapes (points per voxel), so it stays eager and writes into a static buffer
(merged_pool out=); everything after it has static shapes: the voxel block engine, the encoder prior
(uvtr_query_proj), the TensorRT encoder, the sampling decoder (graph_decoder inline: cached bf16
weights), the occupancy branch (grid_sample, projections, fuse, refiner, occ_pred engine) and the map branch (cropper,
map_seg engine). Those are captured once and replayed per frame: no Python, no dispatch, no launch gaps. Same ops, same
order, same dtypes as the eager head, so outputs must be bit-identical (e2e_latency --graph-head checks it on the same
frames).

with trt_student.TrtStudentSplit the adaptor levels come from a side stream; the copy-in waits for it after
pooling (the stand-in's ``wait``), so the LLM branch overlaps the eager front and the pooling.
Graph inputs, copied per frame before replay: the adaptor levels (bf16, small), ego2img (the decoder's camera
matrices); the pooled volume is pooled straight into its static buffer. Per-frame metas reach the captured region
only through those (the rest of img_metas is read as Python constants at capture: dataset_type, img_shape); anything
else frame-dependent would be frozen, which the bit-identity check catches.
Capture blockers handled here: ``Tensor.new_tensor(<python list>)`` is a host -> device copy (upstream builds the
decoder's spatial shapes that way); during warm-up and capture it is served from a cache of tensors with the same
values / dtype / device (created in warm-up, kept alive with the graph).
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from qdistill.query_sampling import ego2img_of


def _is_plain(data) -> bool:
    if isinstance(data, (int, float, bool)):
        return True
    return isinstance(data, (list, tuple)) and all(_is_plain(d) for d in data)


@contextmanager
def _cached_new_tensor(cache: dict):
    """Tensor.new_tensor on plain Python data -> one cached tensor per (data, dtype, device, requires_grad)."""
    f = torch.Tensor.new_tensor

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
        if not _is_plain(data):
            return f(self, data, dtype=dtype, device=device, requires_grad=requires_grad)
        dt, dev = dtype or self.dtype, torch.device(device) if device is not None else self.device
        key = (repr(data), dt, str(dev), requires_grad)
        if key not in cache:
            assert not torch.cuda.is_current_stream_capturing(), f"graph_head: new_tensor({data!r}) first seen in capture"
            cache[key] = f(self, data, dtype=dt, device=dev, requires_grad=requires_grad)
        return cache[key]
    torch.Tensor.new_tensor = new_tensor
    try:
        yield
    finally:
        torch.Tensor.new_tensor = f


def _like(t: torch.Tensor) -> torch.Tensor:
    """Static copy with the input's strides (graph_decoder: strides select GEMM kernels in bf16)."""
    return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device).copy_(t)


class GraphedHead:
    def __init__(self, head, warmup: int = 3):
        self.head, self.warmup = head, warmup
        self.bev = head.bev_modeling
        vt = self.bev.view_trans
        assert hasattr(vt, "_pool") and hasattr(vt, "_block"), \
            "graph_head needs the pool / block split (merged_pool.install install)"
        from qdistill.graph_decoder import GraphedDecoder

        dec = self.bev.head.transformer.decoder
        self.gdec = dec.forward if isinstance(dec.forward, GraphedDecoder) else None
        assert self.gdec is not None, "graph_head: install graph_decoder first (its body runs inline in the graph)"
        self.graphs: dict = {}
        self.new_tensors: dict = {}
        self.captures = 0

    # ------------------------------------------------------------ eager front (upstream modeling_perception.forward)

    def _front(self, img_vit_feats, img_llm_feats):
        bev = self.bev
        if img_llm_feats.dim() == 5:
            B, N, H, W, C = img_llm_feats.shape
            feat_main = img_llm_feats.view(B * N, H, W, C).permute(0, 3, 1, 2)
        else:
            N, H, W, C = img_llm_feats.shape
            B = 1
            feat_main = img_llm_feats.permute(0, 3, 1, 2)
        mlvl = [f.view(B, N, *f.shape[1:]) for f in bev.adaptor(feat_main)]
        if img_vit_feats.dim() == 5:
            Bv, Nv, Hv, Wv, Cv = img_vit_feats.shape
            feat_vit = img_vit_feats.view(Bv * Nv, Hv, Wv, Cv).permute(0, 3, 1, 2)
        else:
            Nv, Hv, Wv, Cv = img_vit_feats.shape
            Bv = 1
            feat_vit = img_vit_feats.permute(0, 3, 1, 2)
        vit = [f.view(Bv, Nv, *f.shape[1:]) for f in bev.vit_neck(feat_vit)]
        depths = [bev.depth_net(f.view(-1, *f.shape[-3:])).softmax(dim=1) for f in vit]
        return mlvl, vit, depths

    # ------------------------------------------------------------ captured region

    def _post(self, st: dict, img_metas):
        bev = self.bev
        y = bev.view_trans._block(st["s"])
        return bev.head(st["mlvl"], img_metas, None, vit_bev_feat=None,
                        uvtr_bev_feat=bev._uvtr_voxel_to_bev_tokens(y), uvtr_occ_feat=y)

    def __call__(self, img_vit_feats, img_llm_feats, img_metas):
        ad = self.bev.adaptor
        wait = getattr(ad, "wait", None)                         # split student: levels filled on a side stream
        if wait is not None:
            ad.defer = True
        mlvl, vit, depths = self._front(img_vit_feats, img_llm_feats)
        key = (tuple((tuple(t.shape), t.stride(), t.dtype) for t in mlvl + vit + depths),
               tuple(int(v) for v in img_metas[0]["img_shape"][0]), img_metas[0].get("dataset_type"))
        if key not in self.graphs:
            if wait is not None:
                wait()                                           # capture reads the levels
            s = self.bev.view_trans._pool(vit, depths, copy.deepcopy(img_metas))   # learn the pooled shape
            self.graphs[key] = self._capture(s, mlvl, img_metas)
        g, st, out = self.graphs[key]
        self.bev.view_trans._pool(vit, depths, copy.deepcopy(img_metas), out=st["s"])
        if wait is not None:
            wait()                                               # the LLM branch ran alongside the front + pooling
            ad.defer = False                                     # eager callers of the same head wait at the adaptor
        for a, b in zip(st["mlvl"], mlvl):
            a.copy_(b)
        st["ego2img"].copy_(ego2img_of(img_metas[0])[0])
        g.replay()
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in out.items()}

    def _capture(self, s, mlvl, img_metas):
        st = {"s": s.contiguous(), "mlvl": [_like(t) for t in mlvl], "ego2img": ego2img_of(img_metas[0])[0].clone()}
        self.gdec.inline, self.gdec.ego2img = True, st["ego2img"]
        try:
            with _cached_new_tensor(self.new_tensors), torch.no_grad():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):                    # warm-up: lazy inits, caches, TRT first enqueue
                    for _ in range(self.warmup):
                        self._post(st, img_metas)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = self._post(st, img_metas)
        finally:
            self.gdec.inline = False                             # the eager path (no graph_head) keeps its own graph
        self.captures += 1
        return g, st, out

    def info(self) -> dict:
        return {"graphs": len(self.graphs), "captures": self.captures, "new_tensor_cache": len(self.new_tensors)}


def install(head) -> GraphedHead:
    """bev_modeling.forward -> eager front + pooling, then the captured post-pool head (in place, permanent)."""
    gh = GraphedHead(head)
    head.bev_modeling.forward = gh                               # replay never runs the decoder's Python
    return gh
