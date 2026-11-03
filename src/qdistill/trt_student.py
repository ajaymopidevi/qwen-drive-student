"""Student + tap de-normalisation + the head's adaptor, neck and depth_net as one static TensorRT engine.

    sf = load("weights/engines/student_front_int8.engine", head)   # installs the stand-ins
    vit, llm = sf(images[None])             # images [1, N_cam, 3, 512, 896], ImageNet-normalised
    run_head(head, vit, llm, img_metas)     # the head reads the engine's outputs; vit and llm are stand-ins

FusedStudentFold folds llm_out + de-normalisation into the adaptor (qdistill.fold_adaptor) and keeps "z", the
LayerNorm'd LLM tokens, as a network output: without it TensorRT's Myelin merges the transformer with the folded adaptor
and stops fusing the attention. FusedStudentFront adds the neck fold and depth_net.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

N_CAM = 6
IMAGES = (1, N_CAM, 3, 512, 896)
VIT = (N_CAM, 32, 56, 1024)
LLM = (N_CAM, 16, 28, 2560)
LEVELS = ((N_CAM, 256, 64, 112), (N_CAM, 256, 32, 56), (N_CAM, 256, 16, 28), (N_CAM, 256, 8, 14))
OUT_NAMES = ("vit", "l0", "l1", "l2", "l3")


class _LN2d(nn.Module):
    """LayerNorm2d (fpn.py) as F.layer_norm over channels: same maths, exports as one LayerNormalization node."""

    def __init__(self, ln):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps
        self.c = int(ln.weight.numel())  # a Python int: the traced x.shape[1] does not export as normalized_shape

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (self.c,), self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


def export_adaptor(adaptor: nn.Module) -> nn.Module:
    """A copy of the SimpleFPN with LayerNorm2d swapped for _LN2d (the head's own module is left as it is)."""
    import copy

    a = copy.deepcopy(adaptor)
    for stage in a.stages:
        for i, m in enumerate(stage):
            if type(m).__name__ == "LayerNorm2d":
                stage[i] = _LN2d(m)
    return a


class FusedStudentFPN(nn.Module):
    """Export graph: images [1, N, 3, 512, 896] -> (vit tap [N, 32, 56, 1024], adaptor levels 0-3)."""

    def __init__(self, student, norm, adaptor):
        super().__init__()
        self.student, self.adaptor = student, export_adaptor(adaptor)
        for name in ("vit", "llm"):
            self.register_buffer(f"{name}_mean", getattr(norm, f"{name}_mean").float().clone())
            self.register_buffer(f"{name}_std", getattr(norm, f"{name}_std").float().clone())

    def forward(self, images):
        vit_n, llm_n = self.student(images)
        vit = vit_n[0] * self.vit_std + self.vit_mean
        llm = llm_n[0] * self.llm_std + self.llm_mean
        return (vit, *self.adaptor(llm.permute(0, 3, 1, 2)))






class FusedStudentFold(nn.Module):
    """export graph: images [1, N, 3, 512, 896] -> (vit tap, adaptor levels 0-3), adaptor folded onto the tokens."""

    def __init__(self, student, norm, adaptor, ln: str = "nhwc"):
        super().__init__()
        from qdistill.fold_adaptor import FoldedAdaptor

        self.student = student
        self.fold = FoldedAdaptor(student.llm_out[1], norm.llm_std.float(), norm.llm_mean.float(), adaptor, ln)
        self.register_buffer("vit_mean", norm.vit_mean.float().clone())
        self.register_buffer("vit_std", norm.vit_std.float().clone())

    def forward(self, images):
        from qdistill.fold_adaptor import llm_tokens

        s = self.student
        p16, z = llm_tokens(s, images)
        vit = s.vit_out(s.vit_neck(p16)).permute(0, 2, 3, 1)  # [N, 32, 56, 1024] (B = 1)
        return (vit * self.vit_std + self.vit_mean, *self.fold(z), z)  # z: a Myelin region boundary (docstring)


class FusedStudentFront(nn.Module):
    """export graph: FusedStudentFold with the head's front inside: images -> (neck [N, 256, 32, 56], depth logits,
    adaptor levels 0-3, z). vit_neck folded onto the student's neck (no 1024-channel tap, no NHWC round trip)."""

    def __init__(self, student, norm, adaptor, vit_neck, depth_net):
        super().__init__()
        from qdistill.fold_adaptor import FoldedAdaptor, FoldedVitNeck

        self.student = student
        self.fold = FoldedAdaptor(student.llm_out[1], norm.llm_std.float(), norm.llm_mean.float(), adaptor)
        self.neck = FoldedVitNeck(student.vit_out, norm.vit_std.float(), norm.vit_mean.float(), vit_neck)
        self.depth_net = depth_net

    def forward(self, images):
        from qdistill.fold_adaptor import llm_tokens

        s = self.student
        p16, z = llm_tokens(s, images)
        neck = self.neck(s.vit_neck(p16))
        return (neck, self.depth_net(neck), *self.fold(z), z)






T512 = (N_CAM, 512, 16, 28)
B_OUT = ("l0", "l1", "l2", "l3")






class _AdaptorStandIn(nn.Module):
    """Installed as bev_modeling.adaptor: ignores its input and returns the engine's levels of this frame."""

    def __init__(self, owner):
        super().__init__()
        self.owner = [owner]  # a list: not a registered submodule

    def forward(self, x):
        lv = self.owner[0].levels
        assert lv is not None, "TrtStudentFPN: run the engine on this frame before the head"
        return tuple(t.to(torch.bfloat16) for t in lv)


class TrtStudentFPN:
    """Runs the fused engine and installs the adaptor stand-in on ``head`` (in place, permanent: the PyTorch adaptor's
    weights are dropped)."""

    def __init__(self, path: str | Path, head):
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.ctx.set_input_shape("images", IMAGES)
        self.out = {}
        for name, shape in zip(OUT_NAMES, (VIT, *LEVELS)):
            self.out[name] = torch.empty(shape, dtype=torch.float16, device="cuda")
            self.ctx.set_tensor_address(name, self.out[name].data_ptr())
        for i in range(self.engine.num_io_tensors):  # extra outputs (fold engine "z") -> scratch
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT and n not in self.out:
                self.out[n] = torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=torch.float16, device="cuda")
                self.ctx.set_tensor_address(n, self.out[n].data_ptr())
        self.levels = None
        self.llm_stand_in = torch.zeros((), dtype=torch.bfloat16, device="cuda").expand(LLM)
        head.bev_modeling.adaptor = _AdaptorStandIn(self)
        self.scratch_MiB = round(self.engine.device_memory_size_v2 / 2**20, 1)

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images [1, N, 3, 512, 896] (any float dtype) -> (vit tap fp16 [N, 32, 56, 1024], llm stand-in)."""
        x16 = images.to(torch.float16).contiguous()
        assert tuple(x16.shape) == IMAGES, (x16.shape, IMAGES)
        self.ctx.set_tensor_address("images", x16.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        self.levels = [self.out[n] for n in OUT_NAMES[1:]]  # read by the stand-in in this frame's head pass
        return self.out["vit"], self.llm_stand_in

    def info(self) -> dict:
        return {"student_fpn_scratch_MiB": self.scratch_MiB}


class _FrontStandIn(nn.Module):
    """installed as bev_modeling.vit_neck / depth_net: ignores its input, returns a bf16 copy of the engine's output."""

    def __init__(self, owner, name, as_list):
        super().__init__()
        self.owner, self.name, self.as_list = [owner], name, as_list

    def forward(self, x):
        y = self.owner[0].out[self.name].to(torch.bfloat16)
        return [y] if self.as_list else y


class TrtStudentFront:
    """the FusedStudentFront engine; same call interface as TrtStudentFPN. Installs stand-ins for the adaptor,
    vit_neck and depth_net on ``head`` (in place, permanent). Head and pooling are unchanged: the softmax stays where it was."""

    def __init__(self, path: str | Path, head):
        import tensorrt as trt

        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.ctx.set_input_shape("images", IMAGES)
        self.out = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.out[n] = torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=torch.float16, device="cuda")
                self.ctx.set_tensor_address(n, self.out[n].data_ptr())
        self.levels = None
        z = dict(dtype=torch.bfloat16, device="cuda")
        self.vit_stand_in, self.llm_stand_in = torch.zeros((), **z).expand(VIT), torch.zeros((), **z).expand(LLM)
        bev = head.bev_modeling
        bev.adaptor = _AdaptorStandIn(self)
        bev.vit_neck, bev.depth_net = _FrontStandIn(self, "neck", True), _FrontStandIn(self, "depth", False)
        self.scratch_MiB = round(self.engine.device_memory_size_v2 / 2**20, 1)

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x16 = images.to(torch.float16).contiguous()
        assert tuple(x16.shape) == IMAGES, (x16.shape, IMAGES)
        self.ctx.set_tensor_address("images", x16.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        self.levels = [self.out[n] for n in OUT_NAMES[1:]]
        return self.vit_stand_in, self.llm_stand_in

    def info(self) -> dict:
        return {"student_front_scratch_MiB": self.scratch_MiB}




def levels_unread(head) -> bool:
    """True when nothing reads the adaptor levels: every encoder layer's spatial cross-attention is head_chain's
    skip_ca and no SamplingDecoder (det_sample) samples them (one reading the ViT neck, cfg levels
    "vit" / "vit3", is fine)."""
    tr = head.bev_modeling.head.transformer
    layers = tr.encoder.layers
    sca = len(layers) > 0 and all(getattr(l.attentions[1], "_qd_skip_ca", False) for l in layers)
    decs = [tr.decoder, getattr(tr.decoder, "dec", None), *head.modules()]
    llm_sampler = any(type(d).__name__ == "SamplingDecoder" and d.cfg.get("levels", "llm") == "llm" for d in decs)
    return sca and not llm_sampler




def load(path: str | Path, head):
    """The runner for a student engine."""
    import tensorrt as trt

    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
    assert eng is not None, path
    names = {eng.get_tensor_name(i) for i in range(eng.num_io_tensors)}
    del eng
    return TrtStudentFront(path, head) if "neck" in names else TrtStudentFPN(path, head)


def self_test() -> dict:
    """CPU: the folds are exact."""
    from types import SimpleNamespace

    from qdistill.student import TapStudent

    torch.manual_seed(0)
    st = TapStudent(pretrained=False).eval()
    norm = SimpleNamespace(vit_mean=torch.randn(1024) * 0.1, vit_std=torch.rand(1024) + 0.5)
    x = torch.randn(1, 1, 3, 512, 896)
    fold = self_test_fold(st, norm, x)
    return fold


def self_test_fold(st, norm, x) -> dict:
    """CPU, float64: FusedStudentFold == FusedStudentFPN (random student, LLM tap statistics and SimpleFPN adaptor)."""
    from qwen_drive_perception.fpn import SimpleFPN

    torch.manual_seed(1)
    ad = SimpleFPN(2560).eval()
    for m in ad.modules():  # non-trivial LayerNorm affines / biases
        if type(m).__name__ == "LayerNorm2d":
            m.weight.data.uniform_(0.5, 1.5); m.bias.data.normal_(0, 0.1)
        if isinstance(m, nn.ConvTranspose2d):
            m.bias.data.normal_(0, 0.1)
    norm.llm_mean, norm.llm_std = torch.randn(2560) * 0.1, torch.rand(2560) + 0.5
    st64, x64 = st.double(), x.double()
    with torch.no_grad():
        ref = FusedStudentFPN(st64, norm, ad.double()).double().eval()(x64)
        got = FusedStudentFold(st64, norm, ad).double().eval()(x64)
        got_c = FusedStudentFold(st64, norm, ad, "nchw").double().eval()(x64)
    rel = [float((a - b).abs().max() / b.abs().max()) for g in (got, got_c) for a, b in zip(g, ref)]
    assert all(r < 1e-6 for r in rel), rel  # the folded weights are stored float32 (~3e-8); a wrong fold is O(1)
    from qdistill.fold_adaptor import FoldedVitNeck, llm_tokens

    neck = SimpleFPN(1024, 256, scale_factors=(1.0,)).eval()
    for m in neck.modules():
        if type(m).__name__ == "LayerNorm2d":
            m.weight.data.uniform_(0.5, 1.5); m.bias.data.normal_(0, 0.1)
    with torch.no_grad():
        h = st64.vit_neck(llm_tokens(st64, x64)[0])
        vit = (st64.vit_out(h).permute(0, 2, 3, 1) * norm.vit_std.double() + norm.vit_mean.double()).permute(0, 3, 1, 2)
        a = neck.double()(vit)[0]
        b = FoldedVitNeck(st64.vit_out, norm.vit_std, norm.vit_mean, neck).double().eval()(h)
    rn = float((a - b).abs().max() / a.abs().max())
    assert rn < 1e-6, rn
    st.float()
    return {"fold_exact_rel": max(rel), "neck_fold_rel": rn}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test()))
