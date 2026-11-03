"""TensorRT runner for the voxel block engine."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class _Voxel(nn.Module):
    """Static fp16 voxel engine [1, 1, X, Y, Z, C] -> [1, C, Z_out, Y, X] (default scratch allocation), output a bf16
    copy. ``z_out``: output levels when the engine changes them (16 in, 8 out); default Z."""

    def __init__(self, path, X, Y, Z, C=256, z_out=None):
        super().__init__()
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.in_shape = (1, 1, X, Y, Z, C)
        self.ctx.set_input_shape("s", self.in_shape)
        self.out = torch.empty((1, C, z_out or Z, Y, X), dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address("y", self.out.data_ptr())
        self.keep_fp16 = False                 # glue_cuts.vol_fp16: return the fp16 buffer itself

    def forward(self, s):
        s = s.to(torch.float16).contiguous()
        assert tuple(s.shape) == self.in_shape, (tuple(s.shape), self.in_shape)
        self.ctx.set_tensor_address("s", s.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out if self.keep_fp16 else self.out.to(torch.bfloat16)
