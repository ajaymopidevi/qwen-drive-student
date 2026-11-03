"""Precompute foreground class maps for the student loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from qdistill.data import split_tokens
from qdistill.fg_weights import fg_class_map, gt_depth_map
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--split", action="append", required=True)
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--output", type=Path, default=None, help="default cache/fg (or cache/gtdepth)/v1.0-trainval")
    p.add_argument("--gt-depth", action="store_true", help="per-patch GT box camera depth (m, 0 = bg) instead")
    args = p.parse_args()
    args.output = args.output or ROOT / ("cache/gtdepth" if args.gt_depth else "cache/fg") / "v1.0-trainval"
    fn = gt_depth_map if args.gt_depth else fg_class_map

    args.output.mkdir(parents=True, exist_ok=True)
    tokens = list(dict.fromkeys(t for s in args.split for t in split_tokens(args.splits, s)))
    todo = [t for t in tokens if not (args.output / f"{t}.npy").exists()]
    for t in tqdm(todo, desc="gt depth maps" if args.gt_depth else "fg maps"):
        np.save(args.output / f"{t}.npy", fn(PerceptionFrame(args.packed / t)).astype(np.float16))
    print(f"{len(todo)} written, {len(tokens) - len(todo)} existed -> {args.output}")


if __name__ == "__main__":
    main()
