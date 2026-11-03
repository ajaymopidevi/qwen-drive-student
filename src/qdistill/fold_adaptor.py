"""the student's llm_out + tap de-normalisation folded into the head's LLM adaptor (SimpleFPN), exactly.

computes, per stride-32 token z (LayerNorm(384) output of the student's LLM transformer):
    llm  = (W z + b) * std + mean          llm_out Linear 384 -> 2560, then de-normalisation: A z + c
    L0   = LN(conv3(LN(1x1(ConvT2(GELU(LN(ConvT1(llm))))))))          ConvT1 2560 -> 1280, ConvT2 1280 -> 640, 1x1 -> 256
    L1   = LN(conv3(LN(1x1(ConvT(llm)))))                              ConvT 2560 -> 1280, 1x1 1280 -> 256
    L2   = LN(conv3(LN(1x1(llm))))                                     1x1 2560 -> 256
    L3   = LN(conv3(LN(1x1(maxpool2(llm)))))
Every map between z and the first LayerNorm of a stage is linear, so (in exact arithmetic):
    L0: ConvT1 o A  -> ConvT 384 -> 1280 (+ a 2 x 2-periodic bias: the folded bias depends on the sub-pixel position);
        1x1 o ConvT2 -> ConvT 1280 -> 256 (uniform bias)
    L1: 1x1 o ConvT o A -> ConvT 384 -> 256 (+ 2 x 2-periodic bias)
    L2: 1x1 o A -> 1x1 384 -> 256 (+ bias)
    L3: max-pool is not linear: A z + c stays (the 2560-ch tap at 16 x 28, as now), then max-pool, 1x1.
Adaptor ~300 -> ~115 GFLOP (6 cameras; the L0 3 x 3 conv at 64 x 112, ~51, is untouched). Folded in float64, stored
float32. LayerNorm2d -> trt_student._LN2d (ln="nhwc": same maths, one ONNX LayerNormalization over a transposed
tensor) or _LN2dC (ln="nchw": upstream's reduce over the channel dim, no transposes; found the
adaptor's LayerNorm / GELU glue ~3 ms, mostly transposes).
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn


class _LN2dC(nn.Module):
    """LayerNorm2d as upstream computes it (mean / var over dim 1 of NCHW)."""

    def __init__(self, ln):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        v = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(v + self.eps) * self.weight[:, None, None] + self.bias[:, None, None]


def _ln(m, mode: str = "nhwc"):
    from qdistill.trt_student import _LN2d
    if type(m).__name__ != "LayerNorm2d":
        return m
    return _LN2d(m) if mode == "nhwc" else _LN2dC(m)


class FoldedAdaptor(nn.Module):
    def __init__(self, llm_out_linear: nn.Linear, llm_std: torch.Tensor, llm_mean: torch.Tensor, adaptor: nn.Module,
                 ln: str = "nhwc"):
        super().__init__()
        assert ln in ("nhwc", "nchw"), ln
        _l = lambda m: _ln(m, ln)
        d = torch.float64
        A = (llm_std.to(d)[:, None] * llm_out_linear.weight.to(d))                     # [2560, 384]
        c = llm_std.to(d) * llm_out_linear.bias.to(d) + llm_mean.to(d)                 # [2560]
        s0, s1, s2, s3 = (list(st) for st in adaptor.stages)
        assert [type(m).__name__ for m in s0] == ["ConvTranspose2d", "LayerNorm2d", "GELU", "ConvTranspose2d", "Conv2d",
                                                  "LayerNorm2d", "Conv2d", "LayerNorm2d"], s0
        assert [type(m).__name__ for m in s1] == ["ConvTranspose2d", "Conv2d", "LayerNorm2d", "Conv2d", "LayerNorm2d"], s1
        assert [type(m).__name__ for m in s2] == ["Conv2d", "LayerNorm2d", "Conv2d", "LayerNorm2d"], s2
        assert [type(m).__name__ for m in s3] == ["MaxPool2d", "Conv2d", "LayerNorm2d", "Conv2d", "LayerNorm2d"], s3

        def convt_a(ct, post=None):  # ConvT(A z + c) [then 1x1 post]: weight [384, O, 2, 2], bias [O, 2, 2]
            w, b = ct.weight.to(d), (ct.bias.to(d) if ct.bias is not None else torch.zeros(ct.out_channels, dtype=d))
            if post is not None:
                p = post.weight.to(d)[:, :, 0, 0]                                     # [O', O]
                assert post.bias is None
                w, b = torch.einsum("pm,cmab->cpab", p, w), p @ b
            wf = torch.einsum("cmab,ck->kmab", w, A)                                  # [384, O, 2, 2]
            bf = torch.einsum("cmab,c->mab", w, c) + b[:, None, None]                 # [O, 2, 2]
            return wf, bf

        # L0
        w, bt = convt_a(s0[0])
        self.l0_ct1 = nn.ConvTranspose2d(A.shape[1], w.shape[1], 2, stride=2, bias=False)
        self.l0_ct1.weight.data = w.float()
        self.register_buffer("l0_b1", bt.float())                                      # [1280, 2, 2]
        self.l0_mid = nn.Sequential(_l(copy.deepcopy(s0[1])), copy.deepcopy(s0[2]))
        ct2, p = s0[3], s0[4]
        pw = p.weight.to(d)[:, :, 0, 0]
        self.l0_ct2 = nn.ConvTranspose2d(ct2.in_channels, pw.shape[0], 2, stride=2, bias=True)
        self.l0_ct2.weight.data = torch.einsum("pm,cmab->cpab", pw, ct2.weight.to(d)).float()
        self.l0_ct2.bias.data = (pw @ ct2.bias.to(d)).float()
        self.l0_tail = nn.Sequential(*[_l(copy.deepcopy(m)) for m in s0[5:]])
        # L1
        w, bt = convt_a(s1[0], s1[1])
        self.l1_ct = nn.ConvTranspose2d(A.shape[1], w.shape[1], 2, stride=2, bias=False)
        self.l1_ct.weight.data = w.float()
        self.register_buffer("l1_b", bt.float())                                       # [256, 2, 2]
        self.l1_tail = nn.Sequential(*[_l(copy.deepcopy(m)) for m in s1[2:]])
        # L2
        q = s2[0].weight.to(d)[:, :, 0, 0]                                             # [256, 2560]
        assert s2[0].bias is None
        self.l2_in = nn.Conv2d(A.shape[1], q.shape[0], 1, bias=True)
        self.l2_in.weight.data = (q @ A).float()[:, :, None, None]
        self.l2_in.bias.data = (q @ c).float()
        self.l2_tail = nn.Sequential(*[_l(copy.deepcopy(m)) for m in s2[1:]])
        # L3: the tap itself (A z + c), max-pool, the stage
        self.l3_tap = nn.Conv2d(A.shape[1], A.shape[0], 1, bias=True)
        self.l3_tap.weight.data = A.float()[:, :, None, None]
        self.l3_tap.bias.data = c.float()
        self.l3 = nn.Sequential(*[_l(copy.deepcopy(m)) for m in s3])

    @staticmethod
    def _tile(b, h, w):  # [O, 2, 2] -> [1, O, 2h, 2w] (the 2 x 2-periodic bias of a folded stride-2 ConvT)
        return b.repeat(1, h, w)[None]

    def forward(self, z):
        """z [N, 384, 16, 28] (the student's LayerNorm'd LLM tokens, NCHW) -> the adaptor's 4 levels."""
        h, w = z.shape[-2:]
        l0 = self.l0_tail(self.l0_ct2(self.l0_mid(self.l0_ct1(z) + self._tile(self.l0_b1, h, w))))
        l1 = self.l1_tail(self.l1_ct(z) + self._tile(self.l1_b, h, w))
        l2 = self.l2_tail(self.l2_in(z))
        l3 = self.l3(self.l3_tap(z))
        return l0, l1, l2, l3


class FoldedVitNeck(nn.Module):
    """the same fold on the ViT path. vit_out (1x1, 1024 ch), de-normalisation and vit_neck's first 1x1 (no bias)
    are linear: one 1x1 from the student's neck to 256 ch, no 1024-channel tensor; vit_neck's LayerNorm + 3x3 + LayerNorm
    stay."""

    def __init__(self, vit_out: nn.Conv2d, vit_std: torch.Tensor, vit_mean: torch.Tensor, vit_neck: nn.Module, ln: str = "nhwc"):
        super().__init__()
        d = torch.float64
        s = list(vit_neck.stages[0])
        assert [type(m).__name__ for m in s] == ["Conv2d", "LayerNorm2d", "Conv2d", "LayerNorm2d"], s
        assert vit_out.kernel_size == (1, 1) and s[0].kernel_size == (1, 1) and s[0].bias is None, (vit_out, s[0])
        A = vit_std.to(d)[:, None] * vit_out.weight.to(d)[:, :, 0, 0]                  # [1024, neck]
        c = vit_std.to(d) * vit_out.bias.to(d) + vit_mean.to(d)                        # [1024]
        P = s[0].weight.to(d)[:, :, 0, 0]                                              # [256, 1024]
        self.conv = nn.Conv2d(A.shape[1], P.shape[0], 1, bias=True)
        self.conv.weight.data = (P @ A).float()[:, :, None, None]
        self.conv.bias.data = (P @ c).float()
        self.tail = nn.Sequential(*[_ln(copy.deepcopy(m), ln) for m in s[1:]])

    def forward(self, h):
        return self.tail(self.conv(h))


def branch_input(student, images):
    """TapStudent's forward up to the LLM branch input: images [1, N, 3, 512, 896] -> (p16, cat[p32, avgpool2(p16)]
    [N, 512, 16, 28])."""
    s = student
    c16, c32 = s.backbone(images.flatten(0, 1))
    p32 = s.lat32(c32)
    p16 = s.lat16(c16) + F.interpolate(p32, size=c16.shape[-2:], mode="nearest")
    return p16, torch.cat([p32, F.avg_pool2d(p16, 2)], dim=1)


def tokens(student, t512, B: int = 1):
    """The LLM branch from its input to the LayerNorm: t512 [B N, 512, 16, 28] -> z [B N, 384, 16, 28]."""
    s = student
    tok = s.llm_in(t512)
    N = tok.shape[0] // B
    h, w = tok.shape[-2:]
    tok = tok.flatten(2).transpose(1, 2) + s.pos_embed
    tok = tok.view(B, N, h * w, -1) + s.cam_embed.weight[:N, None, :]
    tok = s.encoder(tok.reshape(B, N * h * w, -1))
    z = s.llm_out[0](tok)                                                              # LayerNorm(384)
    return z.view(B * N, h, w, -1).permute(0, 3, 1, 2)


def llm_tokens(student, images):
    """TapStudent's forward up to the LLM branch's LayerNorm: images [1, N, 3, 512, 896] -> (p16, z [N, 384, 16, 28])."""
    p16, t512 = branch_input(student, images)
    return p16, tokens(student, t512, images.shape[0])
