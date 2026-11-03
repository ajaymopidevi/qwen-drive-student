"""Pack nuScenes samples into per-frame directories the perception code reads."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from qdistill.json_stream import iter_json_array
from qdistill.nuscenes_pack import NuScenesTables, pack_sample

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", type=Path, default=ROOT / "data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", type=Path, default=None, help="default: data/packed/<version>")
    parser.add_argument("--chunk", type=int, default=100, help="scenes per table pass")
    args = parser.parse_args()

    out = args.output or ROOT / "data/packed" / args.version

    names = sorted(r["name"] for r in iter_json_array(args.dataroot / args.version / "scene.json"))
    n_packed = n_skipped = 0
    for i in range(0, len(names), args.chunk):
        chunk = names[i : i + args.chunk]
        tables = NuScenesTables(args.dataroot, args.version, scene_names=chunk, keyframes_only=True)
        for token in tqdm(tables.samples_in_order(), desc=f"scenes {i + 1}-{i + len(chunk)}/{len(names)}"):
            if (out / token / "frame.json").exists():
                n_skipped += 1
                continue
            pack_sample(tables, token, out)
            n_packed += 1
    print(f"packed {n_packed} samples ({n_skipped} already packed) from {len(names)} scenes into {out}")


if __name__ == "__main__":
    main()
