"""the BEV encoder (6 layers, spatial cross-attention skipped) as one
TensorRT engine, with TensorRT's built-in MultiscaleDeformableAttnPlugin_TRT for the deformable self-attention.

    enc = NoCAEncoder(head)                  # fp32 copy of the (enc_ckpt-loaded) layers; forward(query, pos) bs-first
    export_onnx(enc, path)                   # MSDA -> trt::MultiscaleDeformableAttnPlugin_TRT
    build(path, engine)                      # static shapes, fp16 (layer norms fp32)
    install(head, engine)                    # transformer.get_bev_features -> engine (in place, permanent)
    head_chain "enc_trt:<engine>"           # the same, after the skip_ca / enc_ckpt items

What leaves of an encoder layer: TSA -> norm -> (cross-attention returns its input) -> norm -> FFN -> norm.
Single-frame TSA attends over the BEV queue [query, query] and averages the two outputs. Both queue entries share the
value (the same projected query), so the average equals ONE deformable attention over the queue's 2 x 4 points with
weights halved ("merged", exact in fp32): value_proj runs once on 40,000 cells instead of twice.
Also dropped (unused once cross-attention is gone): the camera projection of the 3D reference points and the image
feature flattening in get_bev_features. Not bit-identical to the bf16 PyTorch path (fp16 engine, merged
attention): checked by output error and a downstream eval.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn

from qwen_drive_perception.attention import multi_scale_deformable_attn_pytorch

H = W = 200
C = 256
HEADS, POINTS, QUEUE = 8, 4, 2


class _MSDA(torch.autograd.Function):
    """value [1, S, M, D], shapes [L, 2] int32, start [L] int32, loc [1, Q, M, L, P, 2], w [1, Q, M, L, P] -> [1, Q, M D].
    PyTorch (fp32 grid_sample) when run; the TensorRT plugin when exported."""

    @staticmethod
    def forward(ctx, value, shapes, start, loc, w):
        return multi_scale_deformable_attn_pytorch(value, shapes.long(), loc, w)

    @staticmethod
    def symbolic(g, value, shapes, start, loc, w):
        out = g.op("trt::MultiscaleDeformableAttnPlugin_TRT", value, shapes, start, loc, w, plugin_version_s="1")
        return out


class _Layer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        tsa = layer.attentions[0]
        assert (tsa.num_heads, tsa.num_points, tsa.num_bev_queue, tsa.num_levels) == (HEADS, POINTS, QUEUE, 1)
        self.value_proj, self.output_proj = tsa.value_proj, tsa.output_proj
        self.sampling_offsets, self.attention_weights = tsa.sampling_offsets, tsa.attention_weights
        self.norms, self.ffn = layer.norms, layer.ffns[0]

    def forward(self, x, pos, ref, shapes, start, normalizer):
        """x, pos [1, Q, C]; ref [1, Q, 1, 2] (x, y in [0, 1]); normalizer [2] = (W, H)."""
        b, q, c = x.shape
        v = self.value_proj(x).view(b, q, HEADS, c // HEADS)
        qc = torch.cat([x, x + pos], -1)
        off = self.sampling_offsets(qc).view(b, q, HEADS, QUEUE, POINTS, 2)
        aw = self.attention_weights(qc).view(b, q, HEADS, QUEUE, POINTS).softmax(-1) * 0.5  # mean over the queue
        loc = ref[:, :, None, :, None, :] + (off / normalizer).reshape(b, q, HEADS, 1, QUEUE * POINTS, 2)
        aw = aw.reshape(b, q, HEADS, 1, QUEUE * POINTS)
        o = _MSDA.apply(v, shapes, start, loc, aw).reshape(b, q, c)
        x = self.norms[0](self.output_proj(o) + x)
        x = self.norms[1](x)  # cross-attention skipped (returns its input)
        return self.norms[2](self.ffn(x))  # FFN adds its own residual


class NoCAEncoder(nn.Module):
    def __init__(self, head, ckpt: str | Path | None = None):
        """``ckpt``: train_enc_recover final.pt loaded in fp32 (weights trained in fp32 must not pass through
        the bf16 head on the way, as head_chain enc_ckpt would round them)."""
        super().__init__()
        enc = head.bev_modeling.head.transformer.encoder
        layers = copy.deepcopy(enc.layers).float()
        if ckpt is not None:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            layers = layers[: ck.get("enc_layers") or len(layers)]  # truncated encoder
            for i, sd in ck["layers"].items():
                layers[int(i)].load_state_dict({k: v.float() for k, v in sd.items()})
        self.layers = nn.ModuleList(_Layer(lay) for lay in layers)
        ref = enc.get_reference_points(H, W, dim="2d", bs=1, device="cpu", dtype=torch.float32)  # [1, HW, 1, 2]
        self.register_buffer("ref", ref)
        self.register_buffer("shapes", torch.tensor([[H, W]], dtype=torch.int32))
        self.register_buffer("start", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("normalizer", torch.tensor([W, H], dtype=torch.float32))

    def forward(self, query, pos):
        """query, pos [1, H W, C] -> BEV embedding [1, H W, C]."""
        x = query
        for lay in self.layers:
            x = lay(x, pos, self.ref, self.shapes, self.start, self.normalizer)
        return x


def export_onnx(enc: NoCAEncoder, path: str | Path, opset: int = 17):
    enc = enc.float().cuda().eval()
    q = torch.randn(1, H * W, C, device="cuda")
    with torch.no_grad():
        torch.onnx.export(enc, (q, q.clone()), str(path), input_names=["query", "pos"], output_names=["bev"],
                          opset_version=opset, custom_opsets={"trt": 1}, dynamo=False)


def build(onnx_path: str | Path, out_path: str | Path, ln_fp32: bool = True, workspace_gb: int = 2,
          precision: str = "fp16") -> dict:
    """Static engine, fp16 I/O. fp16: layer norms (and their reductions) forced to fp32 when ``ln_fp32``.
    bf16: matmul / elementwise /
    softmax / normalization layers constrained to bf16, the plugin runs fp16 / fp32 (its only types)."""
    import tensorrt as trt

    t0 = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(Path(onnx_path).read_bytes()), [parser.get_error(i) for i in range(parser.num_errors)]
    for i in range(network.num_inputs):
        network.get_input(i).dtype = trt.float16
    for i in range(network.num_outputs):  # the dense-head engine has two outputs
        network.get_output(i).dtype = trt.float16
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    n_fp32 = n_bf16 = 0
    if precision == "bf16":
        cfg.set_flag(trt.BuilderFlag.BF16)
        cfg.set_flag(trt.BuilderFlag.FP16)  # lets the plugin pick fp16 over fp32
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        kinds = (trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE, trt.LayerType.SOFTMAX,
                 trt.LayerType.NORMALIZATION, trt.LayerType.ACTIVATION, trt.LayerType.REDUCE)
        for i in range(network.num_layers):
            lay = network.get_layer(i)
            if lay.type in kinds and all(lay.get_output(j).dtype in (trt.float32, trt.float16) for j in range(lay.num_outputs)):
                lay.precision = trt.bfloat16
                for j in range(lay.num_outputs):
                    lay.set_output_type(j, trt.bfloat16)
                n_bf16 += 1
        ln_fp32 = False
    else:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if ln_fp32:
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            lay = network.get_layer(i)
            if lay.type == trt.LayerType.NORMALIZATION:
                lay.precision = trt.float32
                for j in range(lay.num_outputs):
                    lay.set_output_type(j, trt.float32)
                n_fp32 += 1
    engine = builder.build_serialized_network(network, cfg)
    if engine is None:
        return {"failed": True}
    Path(out_path).write_bytes(bytes(engine))
    layers = network.num_layers
    del engine, cfg, network, parser, builder
    return {"engine": str(out_path), "build_s": round(time.time() - t0, 1), "MB": round(Path(out_path).stat().st_size / 2**20, 1),
            "network_layers": layers, "fp32_norm_layers": n_fp32, "bf16_layers": n_bf16, "precision": precision}


class TrtEncoder(nn.Module):
    """Runs the engine: (query, pos) [1, H W, C] any float dtype -> [1, H W, C] in ``out_dtype``."""

    def __init__(self, path: str | Path, out_dtype=torch.bfloat16):
        super().__init__()
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.rt = trt.Runtime(logger)
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.path, self.out_dtype = str(path), out_dtype
        self.buf = {n: torch.empty(1, H * W, C, dtype=torch.float16, device="cuda") for n in ("query", "pos", "bev")}
        for n, t in self.buf.items():
            if n != "bev":
                self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.scratch_MiB = round(self.engine.device_memory_size_v2 / 2**20, 1)

    def forward(self, query, pos):
        self.buf["query"].copy_(query.reshape(1, H * W, C))
        self.buf["pos"].copy_(pos.reshape(1, H * W, C))
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.buf["bev"].to(self.out_dtype)


def install(head, engine: str | Path) -> dict:
    """transformer.get_bev_features -> the engine. Only valid for the encoder (all six cross-attentions
    skipped, the engine built from the same weights)."""
    tr = head.bev_modeling.head.transformer
    run = TrtEncoder(engine)
    tr.trt_encoder = run  # registered: timing hooks can find it

    def get_bev_features(mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=None, **kwargs):
        assert (bev_h, bev_w) == (H, W) and mlvl_feats[0].size(0) == 1, (bev_h, bev_w)
        q = bev_queries.reshape(1, H * W, C)                    # upstream: [HW, C] -> [HW, bs, C] -> bs-first
        pos = bev_pos.flatten(2).permute(0, 2, 1)               # [bs, C, H, W] -> [bs, HW, C]
        return run(q, pos)
    tr.get_bev_features = get_bev_features
    return {"enc_trt": str(engine), "scratch_MiB": run.scratch_MiB}
