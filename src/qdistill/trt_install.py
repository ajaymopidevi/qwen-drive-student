"""Stage 2: run the loaded head with TensorRT engines in place of its voxel-conv stack and occupancy refiner.

    head = load_head(weights)
    trt = TrtHead(head, voxel_engine="weights/engines/voxel_r64_fp16.engine",
                  refiner_engine="weights/engines/occ_refiner_fp16.engine")
    run_head(head, vit, llm, metas)   # engines inside; installed for the life of the process

Engines are the static fp16 ones from trt_voxel_block.py and trt_occ_refiner.py: fp16 in / out, so
the bf16 activations are cast at the boundary. The replaced PyTorch modules are dropped from the head (their weights
are freed). Both engines run one after the other in the same stream, so they share one scratch buffer
(USER_MANAGED device memory, sized for the larger engine): the full head and a separately allocated engine did not
fit 8 GB together in the measure phase.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from qdistill.light_blocks import pooled_volume

VOXEL_IN = (1, 1, 200, 200, 16, 256)    # kernel layout [B, 1, X, Y, Z, C]
VOXEL_OUT = (1, 256, 16, 200, 200)      # [B, C, Z, Y, X]
REFINER_IN = (1, 16, 16, 200, 200)      # [B, C, Z, Y, X]
REFINER_OUT = (1, 32, 16, 200, 200)


class _Engine(nn.Module):
    """One static engine (one fp16 input, one fp16 output) as a module, so forward hooks can time it."""

    def __init__(self, rt, path: Path, in_name: str, in_shape: tuple, out_name: str, out_shape: tuple):
        import tensorrt as trt

        super().__init__()
        self.engine = rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        self.in_name, self.in_shape = in_name, in_shape
        self.ctx.set_input_shape(in_name, in_shape)
        self.out = torch.empty(out_shape, dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address(out_name, self.out.data_ptr())
        self.scratch_bytes = self.engine.device_memory_size_v2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x16 = x.to(torch.float16).contiguous()
        assert tuple(x16.shape) == self.in_shape, (x16.shape, self.in_shape)
        self.ctx.set_tensor_address(self.in_name, x16.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out.to(torch.bfloat16)  # a copy: the output buffer is reused next frame


class TrtHead:
    """Installs the engines on ``head`` (in place, permanent). Either engine may be None (that part stays PyTorch)."""

    def __init__(self, head, voxel_engine: str | Path | None = None, refiner_engine: str | Path | None = None):
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        bev = head.bev_modeling
        self.engines = {}
        if voxel_engine is not None:
            vt = bev.view_trans
            assert list(vt.voxel_shape) == list(VOXEL_IN[2:5]), vt.voxel_shape
            self.voxel = _Engine(self.rt, voxel_engine, "s", VOXEL_IN, "y", VOXEL_OUT)
            self.engines["voxel"] = self.voxel
            vt.conv_layer = nn.ModuleList()  # bypassed below; frees the exact stack's weights

            def forward(mlvl_feats, img_depth, img_metas):
                return self.voxel(pooled_volume(vt, mlvl_feats, img_depth, img_metas))

            vt.forward = forward
        if refiner_engine is not None:
            tr = bev.head.transformer
            self.refiner = _Engine(self.rt, refiner_engine, "x", REFINER_IN, "y", REFINER_OUT)
            self.engines["refiner"] = self.refiner
            tr.occ_decoder = self.refiner  # perception_transformer: occ_decoder(bev_feat) -> [B, 32, Z, Y, X]
        size = max((e.scratch_bytes for e in self.engines.values()), default=0)
        self.scratch = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
        for e in self.engines.values():
            e.ctx.set_device_memory(self.scratch.data_ptr(), size)
        torch.cuda.empty_cache()

    def info(self) -> dict:
        return {"scratch_MiB": round(self.scratch.numel() / 2**20, 1),
                **{f"{k}_scratch_MiB": round(e.scratch_bytes / 2**20, 1) for k, e in self.engines.items()}}
