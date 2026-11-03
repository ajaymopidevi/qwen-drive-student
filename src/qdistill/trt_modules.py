"""the head's remaining eager bf16 modules
as TensorRT fp16 engines, one static engine per module.

    TARGETS                                    # name -> attribute path under head.bev_modeling
    export_onnx(module, x, path)               # fp32 copy; LayerNorm2d -> F.layer_norm (one Normalization node)
    build(onnx, engine)                        # trt_encoder.build: fp16, normalization layers fp32
    install(head, engine_dir, names)           # the modules -> ModEngine (in place, permanent)
    head_chain "mod_trt:<dir>[:<name>+<name>]"
    head_chain "heads_ckpt:<final.pt>"         # trained HEADS weights into the head (head dtype)
    head_chain "heads_fp16:<final.pt>:<a>+<b>" # trained HEADS kept eager, run as trained (fp32 masters, fp16 autocast)

the FRONT modules (frozen, before the encoder) run as
their engines in the training loop; the HEADS are trained as fp32 masters under fp16 autocast (bev2d.Fp16Run) and
saved as final.pt["heads"] = {name: fp32 state dict}; export_onnx(..., sd=) builds their engines from those masters.

Each target is called with one tensor and returns one tensor (vit_neck: a one-element list, SimpleFPN with
scale_factors=(1.0,)). Shapes are static: the engine is built for the shape captured in the deployed pipeline.
Not bit-identical to the eager bf16 path (fp16 engine, fp32 norms): checked per module against fp32 math
(scripts/deployment/trt_modules.py) and downstream.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TARGETS = {
    "vit_neck": "vit_neck",
    "depth_net": "depth_net",
    "uvtr_query_proj": "uvtr_query_proj",
    "uvtr_occ_proj": "head.transformer.uvtr_occ_proj",
    "uvtr_occ_fuse": "head.transformer.uvtr_occ_fuse",
    "occ_pred_head": "head.transformer.occ_pred_head",
    "map_seg": "head.seg_decoder",
}
LIST_OUT = ("vit_neck",)
FRONT = ("vit_neck", "depth_net", "uvtr_query_proj", "uvtr_occ_proj")
HEADS = ("map_seg", "uvtr_occ_fuse", "occ_pred_head")


def get(bev, path: str) -> nn.Module:
    for part in path.split("."):
        bev = getattr(bev, part)
    return bev


def _set(bev, path: str, mod: nn.Module):
    *parent, last = path.split(".")
    setattr(get(bev, ".".join(parent)) if parent else bev, last, mod)


class _LN2d(nn.Module):
    """LayerNorm2d (fpn.py: channel LayerNorm over NCHW, manual mean / var) as F.layer_norm: same math, one ONNX node."""

    def __init__(self, ln):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class _Softplus(nn.Module):
    """nn.Softplus as relu(bx) + log1p(exp(-|bx|)), / b: same values (to ~1e-9 above the threshold), but exp() never
    overflows. ONNX Softplus = log(1 + exp(x)) in an fp16 engine is inf above x ~ 11 (occ_pred_head 934,850
    non-finite outputs)."""

    def __init__(self, sp):
        super().__init__()
        self.beta = sp.beta

    def forward(self, x):
        bx = x * self.beta
        return (F.relu(bx) + torch.log1p(torch.exp(-bx.abs()))) / self.beta


_SWAP = {"LayerNorm2d": _LN2d, "Softplus": _Softplus}


def fp32_copy(module: nn.Module, sd: dict | None = None) -> nn.Module:
    """fp32 copy for export / reference; ``sd``: fp32 master weights to load, else the module's own."""
    m = copy.deepcopy(module).float().eval()
    if sd is not None:
        m.load_state_dict({k: v.float() for k, v in sd.items()})
    for name, child in list(m.named_modules()):
        if type(child).__name__ in _SWAP:
            *parent, last = name.split(".")
            setattr(m.get_submodule(".".join(parent)) if parent else m, last, _SWAP[type(child).__name__](child))
    return m


class _Single(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[0]


def export_onnx(module: nn.Module, name: str, x: torch.Tensor, path: str | Path, opset: int = 17, sd: dict | None = None):
    m = fp32_copy(module, sd).cuda()
    if name in LIST_OUT:
        m = _Single(m)
    with torch.no_grad():
        torch.onnx.export(m, (x.float().cuda(),), str(path), input_names=["x"], output_names=["y"], opset_version=opset,
                          dynamo=False)


class ModEngine(nn.Module):
    """One static fp16 engine in place of a module: input cast to fp16, output (a copy) cast back to the input dtype."""

    def __init__(self, path: str | Path, list_out: bool = False):
        super().__init__()
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        self.rt = trt.Runtime(logger)
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.path, self.list_out = str(path), list_out
        self.in_shape = tuple(self.engine.get_tensor_shape("x"))
        self.ctx.set_input_shape("x", self.in_shape)
        self.out = torch.empty(tuple(self.engine.get_tensor_shape("y")), dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address("y", self.out.data_ptr())
        self.scratch_MiB = round(self.engine.device_memory_size_v2 / 2**20, 1)

    def forward(self, x):
        x16 = x.to(torch.float16).contiguous()
        assert tuple(x16.shape) == self.in_shape, (self.path, tuple(x16.shape), self.in_shape)
        self.ctx.set_tensor_address("x", x16.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        y = self.out.to(x.dtype if x.is_floating_point() else torch.bfloat16, copy=True)  # never the reused buffer
        return [y] if self.list_out else y


def engine_path(engine_dir: str | Path, name: str) -> Path:
    return Path(engine_dir) / f"{name}_fp16.engine"


def install(head, engine_dir: str | Path, names: list[str] | None = None) -> dict:
    """Replace each named module whose engine exists in ``engine_dir`` (default: every target with an engine)."""
    bev = head.bev_modeling
    names = names or [n for n in TARGETS if engine_path(engine_dir, n).exists()]
    done = {}
    for n in names:
        eng = ModEngine(engine_path(engine_dir, n), list_out=n in LIST_OUT)
        _set(bev, TARGETS[n], eng)
        done[n] = {"scratch_MiB": eng.scratch_MiB, "in": eng.in_shape}
    torch.cuda.empty_cache()
    return done


def load_heads(head, ckpt: str | Path) -> list[str]:
    """final.pt["heads"] (fp32 masters) into the head's modules, cast to their dtype (eager use only: export takes the
    fp32 masters via ``sd``)."""
    bev = head.bev_modeling
    heads = torch.load(ckpt, map_location="cpu", weights_only=False)["heads"]
    for n, sd in heads.items():
        m = get(bev, TARGETS[n])
        m = getattr(m, "m", m)  # already an Fp16Run: the fp32 master inside
        dt = next(m.parameters()).dtype
        m.load_state_dict({k: v.to(dt) if v.is_floating_point() else v for k, v in sd.items()})
    return sorted(heads)


def install_fp16_eager(head, ckpt: str | Path, names: list[str]) -> list[str]:
    """Trained HEADS kept eager (no engine) but in their training numerics: bev2d.Fp16Run over the fp32 masters. For
    modules whose engine is slower than eager (uvtr_occ_fuse, input cast cost) yet were trained in fp16."""
    from qdistill.bev2d import Fp16Run

    bev = head.bev_modeling
    heads = torch.load(ckpt, map_location="cpu", weights_only=False)["heads"]
    for n in names:
        _set(bev, TARGETS[n], Fp16Run(fp32_copy(get(bev, TARGETS[n]), heads[n])).cuda().eval())
    return names
