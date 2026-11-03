"""Convert cached teacher taps to the compact training format."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from qdistill import tap_store
from qdistill.np_io import load_npz
from qdistill.data import split_tokens

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    a.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    a.add_argument("--split", required=True)
    a.add_argument("--format", default="vitW+llmInt8", choices=[f for f in tap_store.FORMATS if f != "full"])
    a.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    args = a.parse_args()

    w_file = args.cache / "vit_W.npy"
    if not w_file.exists():
        np.save(w_file, tap_store.head_W(args.weights).numpy())
    W, W_pinv = tap_store.load_W(args.cache)
    out_dir = args.cache / f"taps_{args.format}"
    out_dir.mkdir(exist_ok=True)

    tokens = split_tokens(args.splits, args.split)
    done = written = 0
    t0 = time.perf_counter()
    for i, t in enumerate(tokens):
        dst = out_dir / f"{t}.npz"
        if dst.exists():
            done += 1
            continue
        with load_npz(args.cache / "taps" / f"{t}.npz") as z:
            vit, llm = torch.from_numpy(z["vit"]).float(), torch.from_numpy(z["llm"]).float()
        arrays = tap_store.encode(args.format, vit, llm, W)
        vit_d, llm_d = tap_store.decode(arrays, W_pinv)
        rel = float((vit_d @ W.T - vit @ W.T).norm() / (vit @ W.T).norm())
        if rel > 1e-3 or not torch.isfinite(llm_d).all():
            raise SystemExit(f"{t}: decode check failed (W x rel. error {rel:.2e})")
        tmp = dst.with_name(dst.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(dst)
        written += 1
        if written % 100 == 0:
            print(f"{i + 1}/{len(tokens)} written {written} ({(time.perf_counter() - t0) / written:.2f} s/frame)", flush=True)
    print(f"done: {written} written, {done} already present -> {out_dir}")


if __name__ == "__main__":
    main()
