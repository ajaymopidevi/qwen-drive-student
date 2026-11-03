"""TensorRT helpers for Stage 2: build static-shape engines from ONNX and run them on torch tensors.

INT8 is explicit (Q/DQ nodes): activation ranges are measured in PyTorch on the calibration volumes
(``conv_input_stats``) and written into the ONNX (``insert_qdq``); the builder's implicit calibrator crashed on these
volumes (calibrator.cpp::add, 2026-10-15) and is deprecated in TensorRT 10. TensorRT 10.13 (tensorrt-cu12).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


def calib_subset(files: list, n: int) -> list:
    """n evenly spaced items of ``files`` (12 / 24 / 32 of 32 are nested)."""
    idx = sorted({int(round(i * len(files) / n)) for i in range(n)})
    assert len(idx) == n and idx[-1] < len(files), (n, len(files))
    return [files[i] for i in idx]


CONV = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)


@torch.no_grad()
def conv_input_stats(model: torch.nn.Module, volumes, pct: float = 99.99, sample: int = 1 << 23) -> list[dict]:
    """Per conv call (forward order = the ONNX Conv node order of a trace), per volume: max |x| and the ``pct``
    percentile of |x| (on ``sample`` random elements). ``model`` runs in fp32 on the GPU; ``volumes`` yields inputs."""
    calls, hooks = [], []
    names = {m: n for n, m in model.named_modules()}
    for m in model.modules():
        if isinstance(m, CONV):
            def pre(mod, a):
                x = a[0].detach().abs().flatten().float()
                xs = x[torch.randint(0, x.numel(), (min(sample, x.numel()),), device=x.device)]
                calls.append({"name": names[mod], "w": list(mod.weight.shape), "max": x.max().item(),
                              "pct": torch.quantile(xs, pct / 100).item(), "mse": mse_amax(xs)})
            hooks.append(m.register_forward_pre_hook(pre))
    stats = []
    try:
        for v in volumes:
            calls.clear()
            model(v)
            stats.append(list(calls))
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()
    assert stats and all(len(s) == len(stats[0]) for s in stats)
    return stats


def mse_amax(xs: torch.Tensor, n: int = 96) -> float:
    """Clip value minimising the int8 fake-quant MSE of |x| samples ``xs`` (log-spaced candidates up to the max)."""
    top = xs.max().clamp_min(1e-8)
    best, arg = None, top.item()
    for a in torch.logspace(-3, 0, n, device=xs.device) * top:
        s = a / 127
        err = ((torch.clamp(torch.round(xs / s), max=127) * s - xs) ** 2).mean()
        if best is None or err < best:
            best, arg = err, a.item()
    return arg


def act_amax(stats: list[list[dict]], idx: list[int], method: str) -> list[float]:
    """Per conv input amax over the calibration volumes ``idx``: "max" = max of maxima, "pct" = mean of percentiles
    (TensorRT's percentile calibrator averages per batch the same way), "mse" = mean of per-volume MSE-optimal clips."""
    n = len(stats[0])
    if method == "max":
        return [max(stats[i][k]["max"] for i in idx) for k in range(n)]
    return [float(np.mean([stats[i][k][method] for i in idx])) for k in range(n)]


def insert_qdq(onnx_in: Path, onnx_out: Path, amax: list[float], weight_shapes: list[list[int]], skip: tuple = (),
               fmt: str = "int8"):
    """Explicit INT8: symmetric per-tensor Q/DQ on every Conv's activation input (scale amax / 127) and per-output-
    channel Q/DQ on its weight. Conv k of the graph gets amax[k]; ``weight_shapes`` (from the PyTorch hooks) must
    match the graph's Conv weights in order. Convs in ``skip`` stay unquantized (fp16). ``fmt`` fp8: E4M3 Q/DQ
    (scale amax / 448, opset 19; Ada / Hopper), which keeps relative precision on heavy-tailed activations. The fp8
    graph is converted to FP16 (I/O, conv weights / biases, scales) for a strongly typed build: a weakly typed FP8
    build searched tactics for > 50 min without finishing (2026-10-15). Only valid for graphs whose float tensors
    are the I/O and conv parameters."""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(onnx_in))
    g = m.graph
    qmax = 448.0 if fmt == "fp8" else 127.0
    fdt = np.float16 if fmt == "fp8" else np.float32
    if fmt == "fp8":
        from onnx import TensorProto

        for o in m.opset_import:
            if o.domain in ("", "ai.onnx"):
                o.version = max(o.version, 19)

        def zeros(shape):
            return helper.make_tensor("z", TensorProto.FLOAT8E4M3FN, list(shape), [0.0] * int(np.prod(shape) if shape else 1))
    else:
        def zeros(shape):
            return numpy_helper.from_array(np.zeros(shape, np.int8))
    inits = {i.name: i for i in g.initializer}
    convs = [n for n in g.node if n.op_type == "Conv"]
    assert len(convs) == len(amax) == len(weight_shapes), (len(convs), len(amax), len(weight_shapes))
    conv_params = {i for n in convs for i in n.input[1:]}  # before the Q/DQ rewiring
    nodes = []
    for k, n in enumerate(convs):
        w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        assert list(w.shape) == weight_shapes[k], (k, w.shape, weight_shapes[k])
        if k in skip:
            continue
        a = f"qdq{k}_a"
        ws = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-8) / qmax
        az, wz = zeros(()), zeros((w.shape[0],))
        az.name, wz.name = f"{a}_z", f"qdq{k}_wz"
        g.initializer.extend([
            numpy_helper.from_array(np.array(max(amax[k], 1e-8) / qmax, fdt), f"{a}_s"), az,
            numpy_helper.from_array(ws.astype(fdt), f"qdq{k}_ws"), wz,
        ])
        nodes.append((n, [
            helper.make_node("QuantizeLinear", [n.input[0], f"{a}_s", f"{a}_z"], [f"{a}_q"]),
            helper.make_node("DequantizeLinear", [f"{a}_q", f"{a}_s", f"{a}_z"], [f"{a}_dq"]),
            helper.make_node("QuantizeLinear", [n.input[1], f"qdq{k}_ws", f"qdq{k}_wz"], [f"qdq{k}_wq"], axis=0),
            helper.make_node("DequantizeLinear", [f"qdq{k}_wq", f"qdq{k}_ws", f"qdq{k}_wz"], [f"qdq{k}_wdq"], axis=0),
        ]))
        n.input[0], n.input[1] = f"{a}_dq", f"qdq{k}_wdq"
    if fmt == "fp8":  # FP16 graph: conv parameters and I/O (every other tensor is integer, asserted)
        from onnx import TensorProto

        for i, init in enumerate(list(g.initializer)):
            if init.data_type == TensorProto.FLOAT:
                assert init.name in conv_params, init.name
                g.initializer[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
        for n in g.node:
            if n.op_type == "Constant":
                assert n.attribute[0].t.data_type != TensorProto.FLOAT, n.name
        for v in list(g.input) + list(g.output):
            v.type.tensor_type.elem_type = TensorProto.FLOAT16
        del g.value_info[:]
    order = []
    for n in g.node:  # each Q/DQ chain right before its Conv keeps the graph topologically sorted
        order.extend(next((q for c, q in nodes if c is n), []))
        order.append(n)
    del g.node[:]
    g.node.extend(order)
    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_out))


def build_engine(onnx_path: Path, input_name: str, shape: tuple, out_path: Path, precision: str = "fp16",
                 calib_files: list[Path] | None = None, calib_cache: Path | None = None, workspace_gb: int = 2,
                 opt_level: int | None = None) -> dict:
    """Static-shape engine with fp16 I/O. ``precision``: fp16; int8q = explicit INT8 from a Q/DQ ONNX (fp16 for
    everything without Q/DQ); int8 = implicit entropy calibration on ``calib_files`` (each an fp16 array of
    ``shape``; crashes on the Stage-2 volumes, kept for reference). ``opt_level``: builder optimization level (None =
    TensorRT's default, 3)."""
    import tensorrt as trt

    t0 = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    strong = precision == "fp8q"  # types come from the (FP16 + FP8 Q/DQ) ONNX; precision flags do not apply
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strong else 0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(Path(onnx_path).read_bytes()), [parser.get_error(i) for i in range(parser.num_errors)]
    if not strong:
        for i in range(network.num_inputs):
            network.get_input(i).dtype = trt.float16
        for i in range(network.num_outputs):
            network.get_output(i).dtype = trt.float16
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if opt_level is not None:
        cfg.builder_optimization_level = opt_level
    prof = builder.create_optimization_profile()
    prof.set_shape(input_name, shape, shape, shape)
    cfg.add_optimization_profile(prof)
    if not strong:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8q":
        cfg.set_flag(trt.BuilderFlag.INT8)
    calib = None
    if precision == "int8":
        assert calib_files, "int8 needs calibration files"

        class Calib(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.i = 0
                self.buf = torch.empty(shape, dtype=torch.float16, device="cuda")  # = the network input dtype

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.i >= len(calib_files):
                    return None
                self.buf.copy_(torch.from_numpy(np.load(calib_files[self.i])).reshape(shape))
                self.i += 1
                return [int(self.buf.data_ptr())]

            def read_calibration_cache(self):
                return calib_cache.read_bytes() if calib_cache and calib_cache.exists() else None

            def write_calibration_cache(self, cache):
                if calib_cache:
                    calib_cache.write_bytes(bytes(cache))

        calib = Calib()
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calib
        cfg.set_calibration_profile(prof)
    engine = builder.build_serialized_network(network, cfg)
    if engine is None:
        return {"failed": True}
    Path(out_path).write_bytes(bytes(engine))
    del engine, cfg, network, parser, builder, calib
    torch.cuda.empty_cache()
    return {"engine": str(out_path), "build_s": round(time.time() - t0, 1), "MB": round(Path(out_path).stat().st_size / 2**20, 1),
            "calib_files": [Path(f).name for f in calib_files] if precision == "int8" else None}


class TrtRunner:
    """One-input / one-output static engine on torch tensors (fp16 in, fp16 out)."""

    def __init__(self, path: Path, input_name: str, in_shape: tuple, output_name: str, out_shape: tuple):
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.input_name = input_name
        self.ctx.set_input_shape(input_name, in_shape)
        self.out = torch.empty(out_shape, dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address(output_name, self.out.data_ptr())

    def __call__(self, x16: torch.Tensor) -> torch.Tensor:
        assert x16.dtype == torch.float16 and x16.is_contiguous()
        self.ctx.set_tensor_address(self.input_name, x16.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def median_ms(fn, n: int = 100, warm: int = 10) -> tuple[float, float]:
    """(median, p90) wall-clock ms of ``fn`` with device syncs."""
    ms = []
    for k in range(warm + n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if k >= warm:
            ms.append(1000 * (time.perf_counter() - t))
    return float(np.median(ms)), float(np.percentile(ms, 90))


def time_engine(path: Path, prof_shapes: dict | None = None) -> list:
    """Median and p90 ms of an engine on random input."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    eng = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
    assert eng is not None, path
    ctx = eng.create_execution_context()
    keep = []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            sh = tuple(eng.get_tensor_shape(n))
            if -1 in sh:
                sh = tuple((prof_shapes or {}).get(n) or eng.get_tensor_profile_shape(n, 0)[1])
            t = torch.rand(sh, device="cuda").half()
            ctx.set_input_shape(n, sh)
            ctx.set_tensor_address(n, t.data_ptr())
            keep.append(t)
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            t = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=torch.float16, device="cuda")
            ctx.set_tensor_address(n, t.data_ptr())
            keep.append(t)
    stream = torch.cuda.current_stream().cuda_stream
    return list(median_ms(lambda: ctx.execute_async_v3(stream), n=100))
