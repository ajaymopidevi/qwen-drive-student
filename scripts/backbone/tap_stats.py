"""Per-channel mean and std of the teacher taps on a training split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from qdistill.data import split_tokens

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-mini")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-mini.json")
    parser.add_argument("--split", default="mini_train")
    args = parser.parse_args()

    tokens = split_tokens(args.splits, args.split)
    acc = {}
    for token in tqdm(tokens, desc="stats"):
        taps = np.load(args.cache / "taps" / f"{token}.npz")
        for name in ("vit", "llm"):
            x = torch.from_numpy(taps[name]).double().reshape(-1, taps[name].shape[-1])
            s = acc.setdefault(name, {"n": 0, "sum": 0.0, "sq": 0.0, "absmax": 0.0})
            s["n"] += x.shape[0]
            s["sum"] = s["sum"] + x.sum(0)
            s["sq"] = s["sq"] + (x * x).sum(0)
            s["absmax"] = torch.maximum(torch.as_tensor(s["absmax"]), x.abs().amax(0))

    out = {}
    for name, s in acc.items():
        mean = s["sum"] / s["n"]
        std = (s["sq"] / s["n"] - mean**2).clamp_min(1e-12).sqrt()
        out[f"{name}_mean"] = mean.float().numpy()
        out[f"{name}_std"] = std.float().numpy()
        ratio = (s["absmax"] / std).numpy()
        print(f"{name}: tokens {s['n']}, channels {mean.numel()}, std median {std.median():.3f} "
              f"[min {std.min():.3g}, max {std.max():.3g}], |x|max/std median {np.median(ratio):.1f} max {ratio.max():.1f}")
    path = args.cache / f"tap_stats_{args.split.replace(':', '_')}.npz"
    np.savez(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
