"""the sampling decoder replayed as a CUDA graph.

    from qdistill import graph_decoder
    graph_decoder.install(head)            # after query_sampling.install / head_chain det_sample (inference only)

the decoder runs ~10 ms of kernels in ~48 ms (~820 launches per frame, incl. the per-frame bf16 cast of its
fp32 master weights). Here the weights are cast once, the frame's tensors are copied into static buffers, and the
captured decoder (6 layers + samplers + prepare_value's concatenation) is replayed. Same ops, same order, same dtypes
as SamplingDecoder.forward, so outputs should be bit-identical (checked by e2e_latency --graph-decoder's agreement).
One graph per (dtype, image size, input shapes and strides); the camera matrices (ego2img) are an input, copied per frame.
Inference only: the cached bf16 weights are not refreshed if the decoder is trained afterwards.
"""

from __future__ import annotations

import os

import torch

from qdistill.query_sampling import SamplingDecoder, ego2img_of, prepare_value


def _like(t: torch.Tensor) -> torch.Tensor:
    """Static copy with the input's strides: query / query_pos arrive as stride-512 views of the query embedding, and
    a compact copy makes the bf16 GEMMs pick other kernels (decoder outputs up to 1.9 apart after 6 layers)."""
    return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device).copy_(t)


class GraphedDecoder:
    def __init__(self, dec: SamplingDecoder, warmup: int = 3):
        self.dec, self.warmup = dec, warmup
        self.graphs: dict = {}
        self.weights: dict = {}
        self.captures = 0
        # graph_head: run inline (no own graph; the caller captures the whole post-pool head), ego2img from this static
        # buffer instead of the frame's metas (a host -> device copy cannot be captured)
        self.inline = False
        self.ego2img: torch.Tensor | None = None
        self.consts: dict = {}

    def _weights(self, dtype):
        if dtype not in self.weights:
            core = self.dec.core
            self.weights[dtype] = {**{k: v.detach().to(dtype) for k, v in core.named_parameters()},
                                   **{k: v for k, v in core.named_buffers()}}
        return self.weights[dtype]

    def _run(self, st: dict, consts: dict):
        """SamplingDecoder.forward's body on the static buffers (prepare_value's concatenation inside the graph)."""
        dtype, g = consts["dtype"], self.dec.cfg["groups"]
        levels = [f.to(dtype) for f in st["levels"]]
        n_cam, c = levels[0].shape[:2]
        v = torch.cat([f[i].flatten(1).T for i in range(n_cam) for f in levels], 0).contiguous().view(1, -1, g, c // g)
        with torch.autocast("cuda", enabled=False):
            return torch.func.functional_call(
                self.dec.core, self._weights(dtype),
                (st["query"], st["value"], st["query_pos"], st["ref"], consts["reg_branches"], st["shapes"],
                 st["start"], (v, consts["vshapes"], consts["vstart"]), st["ego2img"], consts["hw"], consts["kw"]))

    def __call__(self, query, value, query_pos, reference_points, reg_branches=None, spatial_shapes=None,
                 level_start_index=None, **kwargs):
        assert self.dec.levels is not None, "GraphedDecoder: the levels hook (query_sampling.install) did not run"
        kw = dict(kwargs)
        kw.pop("key", None)
        bad = [k for k, v in kw.items() if torch.is_tensor(v)]
        assert not bad, f"GraphedDecoder: tensor kwargs would be frozen into the graph: {bad}"
        if self.inline:
            return self._inline(query, value, query_pos, reference_points, reg_branches, spatial_shapes,
                                level_start_index, kw)
        ego2img, hw = ego2img_of(kw["img_metas"][0])
        cur = {"query": query, "value": value, "query_pos": query_pos, "ref": reference_points,
               "shapes": spatial_shapes, "start": level_start_index, "levels": list(self.dec.levels), "ego2img": ego2img}
        key = (value.dtype, hw, id(reg_branches),
               tuple((k, tuple(t.shape), t.stride(), t.dtype) for k, t in cur.items() if torch.is_tensor(t)),
               tuple((tuple(t.shape), t.stride(), t.dtype) for t in cur["levels"]))
        with torch.no_grad():
            if key not in self.graphs:
                self.graphs[key] = self._capture(cur, value.dtype, hw, reg_branches, kw)
            g, st, out, _ = self.graphs[key]
            for k, t in cur.items():
                if k == "levels":
                    for s, x in zip(st[k], t):
                        s.copy_(x)
                else:
                    st[k].copy_(t)
            g.replay()
            res = tuple(o.clone() for o in out)
            if os.environ.get("QD_GDEC_CHECK"):  # debug: eager decoder on the same inputs
                ref = SamplingDecoder.forward(self.dec, query, value, query_pos, reference_points, reg_branches,
                                              spatial_shapes, level_start_index, **kwargs)
                print("gdec check:", [float((a.float() - b.float()).abs().max()) for a, b in zip(res, ref)], flush=True)
            return res

    def _capture(self, cur, dtype, hw, reg_branches, kw):
        st = {k: ([_like(x) for x in t] if k == "levels" else _like(t)) for k, t in cur.items()}
        _, vshapes, vstart = prepare_value([f.to(dtype) for f in st["levels"]], self.dec.cfg["groups"])
        consts = {"dtype": dtype, "hw": hw, "reg_branches": reg_branches, "kw": kw, "vshapes": vshapes, "vstart": vstart}
        self._weights(dtype)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm-up: lazy inits, constant caches, cuBLAS workspaces
            for _ in range(self.warmup):
                self._run(st, consts)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._run(st, consts)
        self.captures += 1
        return g, st, out, consts  # consts keeps vshapes / vstart (read by the captured kernels) alive

    def _inline(self, query, value, query_pos, reference_points, reg_branches, spatial_shapes, level_start_index, kw):
        """The same body on the caller's tensors (graph_head warm-up and capture); constants built on first use,
        which is a warm-up run outside the capture."""
        assert self.ego2img is not None, "GraphedDecoder inline: graph_head did not set the ego2img buffer"
        h, w = kw["img_metas"][0]["img_shape"][0]
        cur = {"query": query, "value": value, "query_pos": query_pos, "ref": reference_points, "shapes": spatial_shapes,
               "start": level_start_index, "levels": list(self.dec.levels), "ego2img": self.ego2img}
        key = (value.dtype, (int(h), int(w)), id(reg_branches), tuple((tuple(t.shape), t.dtype) for t in cur["levels"]))
        if key not in self.consts:
            assert not torch.cuda.is_current_stream_capturing(), "GraphedDecoder inline: no warm-up before capture"
            _, vshapes, vstart = prepare_value([f.to(value.dtype) for f in cur["levels"]], self.dec.cfg["groups"])
            self.consts[key] = {"dtype": value.dtype, "hw": (int(h), int(w)), "reg_branches": reg_branches, "kw": kw,
                                "vshapes": vshapes, "vstart": vstart}
            self._weights(value.dtype)
        return self._run(cur, self.consts[key])

    def info(self) -> dict:
        return {"graphs": len(self.graphs), "captures": self.captures}


def install(head) -> GraphedDecoder:
    """Replace the installed SamplingDecoder's forward with the graphed one (in place, permanent)."""
    dec = head.bev_modeling.head.transformer.decoder
    assert isinstance(dec, SamplingDecoder), f"graph_decoder needs the SamplingDecoder, found {type(dec).__name__}"
    gd = GraphedDecoder(dec)
    dec.forward = gd
    return gd
