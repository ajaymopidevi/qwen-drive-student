"""Assemble the deployed head from a chain spec: a comma list of items applied in order to the teacher's head.

    from qdistill import head_chain
    head_chain.install(head, "skip_ca:0,enc_ckpt:weights/fine_tune.pt,...")

Items:
  enc_ckpt:<path>     encoder layers from a fine-tune checkpoint
  heads_ckpt:<path>   trained heads from a fine-tune checkpoint; the UVTR projections stay eager in fp16
  skip_ca:<l>         encoder layer l reads no images (its spatial cross-attention returns the query)
  warp_grid:<b>:<block engine>[:<ckpt>]  warped BEV grid, fine within +-b m (qdistill.warp_grid)
  warp_offsets        decoder sampling offsets in metric units on the warped grid
  zhalf:<engine>      voxel block at 8 height levels (qdistill.zhalf)
  zhalf2:<engine>[:fold]  the same block on the warped grid, with the learned height compression; fold = exact folds
  enc_trt:<engine>    BEV encoder as a TensorRT engine
  mod_trt:<dir>:<a>+<b>  named head modules as TensorRT engines
  heads_fp16:<path>:<a>+<b>  named trained heads eager under fp16 autocast
  dense_trt:<engine>  occupancy and map heads as one TensorRT engine (qdistill.dense_engine)
  pool_fold           exact fold of the pooling output
  last_out            only the last decoder layer's outputs (qdistill.glue_cuts)
  vol_fp16            the query projection and dense engine read the block's fp16 volume (qdistill.glue_cuts)
  fast_geom           GEMM coordinate preparation (qdistill.fast_geom)
  det_sample:<path>   decoder queries sample image levels in every camera (qdistill.query_sampling)
"""

from __future__ import annotations

import sys

import torch


def _log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def install(head, spec: str) -> list[str]:
    bev = head.bev_modeling
    layers = bev.head.transformer.encoder.layers
    done = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        kind, _, arg = item.partition(":")
        if kind == "enc_ckpt":
            ck = torch.load(arg, map_location="cpu", weights_only=False)
            for i, sd in ck["layers"].items():
                lay = layers[int(i)]
                lay.load_state_dict({k: v.to(next(lay.parameters()).dtype) for k, v in sd.items()})
        elif kind == "warp_grid":
            from qdistill import warp_grid
            b, _, rest = arg.partition(":")
            eng, _, ck = rest.partition(":")
            _log("warp grid:", warp_grid.install(head, float(b), eng, ck or None))
        elif kind == "warp_offsets":
            from qdistill import warp_grid
            _log("warp offsets:", warp_grid.enable_offsets(head), "decoder attentions")
        elif kind == "zhalf":
            from qdistill import zhalf
            _log("zhalf block:", zhalf.install(head, arg))
        elif kind == "zhalf2":
            from qdistill import zhalf
            eng, _, opt = arg.partition(":")
            _log("zhalf block:", zhalf.install2(head, eng, fold=opt == "fold"))
        elif kind == "enc_trt":
            from qdistill import trt_encoder
            _log("TensorRT encoder:", trt_encoder.install(head, arg))
        elif kind == "mod_trt":
            from qdistill import trt_modules
            d, _, names = arg.partition(":")
            _log("module engines:", trt_modules.install(head, d, names.split("+") if names else None))
        elif kind == "heads_fp16":
            from qdistill import trt_modules
            ck, _, names = arg.partition(":")
            _log("trained heads eager in fp16:", trt_modules.install_fp16_eager(head, ck, names.split("+")))
        elif kind == "heads_ckpt":
            from qdistill import trt_modules
            names = trt_modules.load_heads(head, arg)
            _log("trained heads loaded:", names)
            proj = [n for n in ("uvtr_query_proj", "uvtr_occ_proj") if n in names]
            if proj:  # trained under fp16 autocast; zhalf.install_fold folds them
                _log("UVTR projections eager in fp16:", trt_modules.install_fp16_eager(head, arg, proj))
        elif kind == "pool_fold":
            from qdistill import warp_grid
            _log("pooling fold:", warp_grid.fold_pool(head))
        elif kind == "dense_trt":
            from qdistill import dense_engine
            _log("dense-head engine:", dense_engine.install(head, arg))
        elif kind == "last_out":
            from qdistill import glue_cuts
            _log("last-layer outputs:", glue_cuts.last_out(head))
        elif kind == "vol_fp16":
            from qdistill import glue_cuts
            _log("fp16 volume:", glue_cuts.vol_fp16(head))
        elif kind == "fast_geom":
            from qdistill import fast_geom
            _log("fast geometry:", fast_geom.install(head, arg or "einsum"))
        elif kind == "det_sample":
            from qdistill import query_sampling
            _log("sampling decoder:", query_sampling.install(head, arg))
        elif kind == "skip_ca":
            att = layers[int(arg)].attentions[1]
            att.forward = lambda query, *a, **k: query
            att._qd_skip_ca = True
        else:
            raise ValueError(f"unknown chain item {item!r}")
        done.append(item)
    return done
