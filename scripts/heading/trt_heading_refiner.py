"""Export and build the heading refiner TensorRT engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_heading_refiner import MEAN, STD, HeadingRefiner

ROOT = Path(__file__).resolve().parents[2]


class BothClasses(nn.Module):
    def __init__(self, m: HeadingRefiner):
        super().__init__()
        self.m = m

    def forward(self, x):
        f = self.m.backbone(x).flatten(1)  # x already ImageNet-normalised (outside: fp16 overflow, see below)
        e = self.m.cls.weight                                                    # [2, 32]
        o = [self.m.out(torch.cat([f, torch.ones_like(f[:, :1]) * e[k]], 1)) for k in range(2)]
        return torch.cat(o, 1)


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = HeadingRefiner(pretrained=False).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("export", "build", "measure"), required=True)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/heading_refiner.pt")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/engines")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--crops", type=Path, default=ROOT / "outputs/heading_crops/pred")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / "heading_refiner.onnx"
    eng = args.output / f"heading_refiner_fp16_b{args.batch}.engine"

    if args.phase == "export":
        m, _ = load(args.ckpt)
        w = BothClasses(m).eval()
        x = (torch.rand(2, 3, 128, 64) * 255 - MEAN) / STD
        with torch.no_grad():
            o = w(x)
            for k in range(2):
                a, c = m(x * STD + MEAN, torch.full((2,), k))
                assert torch.allclose(torch.atan2(o[:, 3 * k], o[:, 3 * k + 1]), a, atol=1e-5)
                assert torch.allclose(o[:, 3 * k + 2], c, atol=1e-5)
        torch.onnx.export(w, (x,), str(onnx_path), input_names=["x"], output_names=["y"], opset_version=17,
                          dynamic_axes={"x": {0: "B"}, "y": {0: "B"}}, dynamo=False)
        print(f"export ok (both-class wrapper = model) -> {onnx_path}")
    elif args.phase == "build":
        from qdistill.trt_utils import build_engine
        r = build_engine(onnx_path, "x", (args.batch, 3, 128, 64), eng, precision="fp16")
        print(json.dumps(r, default=str))
    else:
        from qdistill.heading_crops import wrap
        from qdistill.trt_utils import TrtRunner, median_ms
        m, ck = load(args.ckpt)
        m = m.cuda()
        run = TrtRunner(eng, "x", (args.batch, 3, 128, 64), "y", (args.batch, 6))
        xs = sorted(args.crops.glob("pred_*_x.npy"))[:2]
        x = np.concatenate([np.load(f) for f in xs])
        meta = [np.load(str(f).replace("_x.npy", "_meta.npz")) for f in xs]
        cls = np.concatenate([mm["cls"] for mm in meta]); a_dec = np.concatenate([mm["alpha"] for mm in meta])
        ci = torch.tensor([0 if c == 4 else 1 for c in cls]).cuda()
        A_pt, C_pt, A_trt, C_trt = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(x) - args.batch + 1, args.batch):
                xb = torch.from_numpy(x[i:i + args.batch]).cuda().permute(0, 3, 1, 2).float()
                cb = ci[i:i + args.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a, c = m(xb, cb)
                A_pt.append(a.float().cpu()); C_pt.append(torch.sigmoid(c.float()).cpu())
                o = run(((xb - MEAN.cuda()) / STD.cuda()).half().contiguous()).float().view(-1, 2, 3)[torch.arange(len(cb)), cb]
                A_trt.append(torch.atan2(o[:, 0], o[:, 1]).cpu()); C_trt.append(torch.sigmoid(o[:, 2]).cpu())
        A_pt, C_pt, A_trt, C_trt = (torch.cat(t).numpy() for t in (A_pt, C_pt, A_trt, C_trt))
        n = len(A_pt)
        tau = np.where(cls[:n] == 4, ck["eval"]["tau"], ck["eval"]["tau_bike"])
        fp = (np.abs(wrap(A_pt - a_dec[:n])) > np.pi / 2) & (C_pt >= tau)
        ft = (np.abs(wrap(A_trt - a_dec[:n])) > np.pi / 2) & (C_trt >= tau)
        xf = torch.from_numpy(x[:args.batch]).cuda().permute(0, 3, 1, 2).float()
        xb = ((xf - MEAN.cuda()) / STD.cuda()).half().contiguous()
        with torch.no_grad():
            def pt():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(xf, ci[:args.batch])
            t_pt, t_trt = median_ms(pt), median_ms(lambda: run(xb))
        res = {"crops": n, "alpha_abs_diff_deg_median": float(np.degrees(np.median(np.abs(wrap(A_pt - A_trt))))),
               "alpha_abs_diff_deg_p99": float(np.degrees(np.percentile(np.abs(wrap(A_pt - A_trt)), 99))),
               "conf_abs_diff_max": float(np.abs(C_pt - C_trt).max()),
               "flip_decisions_differ": int((fp != ft).sum()), "flips_pt": int(fp.sum()), "flips_trt": int(ft.sum()),
               f"latency_b{args.batch}_ms": {"pytorch_bf16": t_pt, "trt_fp16": t_trt}}
        (args.output / "trt_heading_measure.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
