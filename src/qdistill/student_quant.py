"""INT8 student backbone in PyTorch: the same fake quantization the Q/DQ ONNX applies (trt_utils.insert_qdq), for
calibration choice (PTQ) and quantization-aware training (QAT).

insert_qdq on conv k: activation input symmetric per-tensor, scale amax / 127, values clipped to +-127 steps; weight
per output channel, scale max|w_c| / 127. QuantConv does exactly this in front of F.conv2d. The fold ONNX has no
BatchNormalization: the exporter folds BN into the convs, so the deployed weights are w_c * k_c. A per-channel scale
follows k_c, so the quantized grid is the same relative to the channel and fake-quantizing the unfolded weight here is
the same operation (self_test). BN stays in eval mode; amax is fixed (QAT trains the weights to the deployed ranges).

    python -m qdistill.student_quant    # CPU self-test
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

QMAX = 127.0


def fake_quant(x: torch.Tensor, scale) -> torch.Tensor:
    """round(x / scale) clipped to +-127, times scale; gradient straight through."""
    q = torch.clamp(torch.round(x / scale), -QMAX, QMAX) * scale
    return x + (q - x).detach()


def weight_scale(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel scale of insert_qdq, shaped for broadcasting over ``w``."""
    s = torch.clamp_min(w.detach().abs().flatten(1).max(1).values, 1e-8) / QMAX
    return s.view(-1, *([1] * (w.dim() - 1)))


class QuantConv(nn.Module):
    """A Conv2d with insert_qdq's Q/DQ on its input and weight. Shares the conv's parameters."""

    def __init__(self, conv: nn.Conv2d, amax: float):
        super().__init__()
        assert conv.padding_mode == "zeros" and not isinstance(conv.padding, str), conv
        self.conv, self.enabled = conv, True
        self.register_buffer("scale", torch.tensor(max(amax, 1e-8) / QMAX), persistent=False)

    def forward(self, x):
        c = self.conv
        w = c.weight
        if self.enabled:
            w = fake_quant(w, weight_scale(w))
            x = fake_quant(x.float(), self.scale.to(x.device))
        return F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)


def onnx_to_module(name: str) -> str:
    """"/backbone/stages_3/blocks/blocks.0/aggregation/aggregation.1/conv/Conv" -> "backbone.stages_3.blocks.0.aggregation.1.conv"."""
    parts = name.strip("/").split("/")
    assert parts[-1] == "Conv", name
    out = []
    for i, p in enumerate(parts[:-1]):  # a Sequential's scope repeats its name: blocks/blocks.0; layers.0 has no parent scope
        prev = parts[i - 1] if i else ""
        out.append(p[len(prev) + 1:] if prev and p.startswith(prev + ".") else p)
    return ".".join(out)


def share_key(name: str, level: str) -> str:
    """the scale group of a backbone conv. block: backbone.stem / backbone.stages_N.downsample /
    backbone.stages_N.blocks.K; stage: backbone.stem / backbone.stages_N; global: one group."""
    p = name.split(".")
    if level == "global":
        return "backbone"
    if level == "stage" or p[1] == "stem":
        return ".".join(p[:2])
    return ".".join(p[:4] if p[2] == "blocks" else p[:3])


def shared_amax(amax: dict[str, float], level: str, agg: str) -> dict[str, float]:
    """One amax per group of ``level``: the group's max (no extra clipping) or median of the per-conv ranges."""
    groups: dict[str, list[float]] = {}
    for n, a in amax.items():
        groups.setdefault(share_key(n, level), []).append(a)
    f = {"max": max, "median": lambda v: float(torch.tensor(v).median())}[agg]
    return {n: f(groups[share_key(n, level)]) for n in amax}


def install(student: nn.Module, amax: dict[str, float]) -> None:
    """Wrap the convs named in ``amax`` (module paths under ``student``) in QuantConv, in place."""
    for name, a in amax.items():
        parent, _, leaf = name.rpartition(".")
        p, m = student.get_submodule(parent), student.get_submodule(name)
        assert type(m) is nn.Conv2d or isinstance(m, nn.Conv2d), (name, type(m))
        setattr(p, leaf, QuantConv(m, a).to(m.weight.device))


def remove(student: nn.Module) -> None:
    """Undo ``install``: the state dict is the plain student's again."""
    for name, m in list(student.named_modules()):
        if isinstance(m, QuantConv):
            parent, _, leaf = name.rpartition(".")
            setattr(student.get_submodule(parent), leaf, m.conv)


def self_test(onnx_path=None) -> dict:
    """CPU. Known-answer fake quant + gradient, BN-fold invariance of the weight grid, install / remove exact, ONNX names."""
    from qdistill.student import TapStudent

    x = torch.tensor([0.25, 0.35, -0.25, 20.0, -20.0], requires_grad=True)
    q = fake_quant(x, 0.1)
    assert torch.allclose(q, torch.tensor([0.2, 0.4, -0.2, 12.7, -12.7]), atol=1e-5), q  # half to even, as QuantizeLinear
    q.sum().backward()
    assert torch.equal(x.grad, torch.ones(5)), x.grad

    torch.manual_seed(0)
    w, k = torch.randn(8, 4, 3, 3), torch.rand(8).view(-1, 1, 1, 1) + 0.2
    rel = float((fake_quant(w * k, weight_scale(w * k)) - fake_quant(w, weight_scale(w)) * k).abs().max() / (w * k).abs().max())
    assert rel < 1e-5, rel

    st = TapStudent(pretrained=False).eval()
    keys = list(st.state_dict())
    names = [n for n, m in st.named_modules() if isinstance(m, nn.Conv2d) and n.startswith("backbone.")]
    img = torch.randn(1, 1, 3, 512, 896)
    with torch.no_grad():
        ref = st.backbone(img.flatten(0, 1))
        install(st, {n: 4.0 for n in names})
        for m in st.modules():
            if isinstance(m, QuantConv):
                m.enabled = False
        off = st.backbone(img.flatten(0, 1))
        for m in st.modules():
            if isinstance(m, QuantConv):
                m.enabled = True
        on = st.backbone(img.flatten(0, 1))
        remove(st)
        back = st.backbone(img.flatten(0, 1))
    assert all(torch.equal(a, b) for a, b in zip(ref, off)), "disabled QuantConv differs from the conv"
    assert all(torch.equal(a, b) for a, b in zip(ref, back)), "remove() does not restore the backbone"
    assert not all(torch.equal(a, b) for a, b in zip(ref, on)), "quantization had no effect"
    assert list(st.state_dict()) == keys, "remove() changed the state dict keys"
    res = {"backbone_convs": len(names), "quant_changes_output": True}
    if onnx_path is not None:
        import onnx

        m = onnx.load(str(onnx_path), load_external_data=False)
        got = [onnx_to_module(n.name) for n in m.graph.node if n.op_type == "Conv" and n.name.startswith("/backbone/")]
        assert sorted(got) == sorted(names), sorted(set(got) ^ set(names))[:6]
        res["onnx_names_match"] = len(got)
    return res


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(self_test(sys.argv[1] if len(sys.argv) > 1 else None)))
