"""Ego poses per sample token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdistill.json_stream import iter_json_array
from qdistill.nuscenes_pack import NuScenesTables

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", type=Path, default=ROOT / "data/nuscenes")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--split", default="val")
    p.add_argument("--extra-scenes", nargs="*", default=[])
    p.add_argument("--output", type=Path, default=ROOT / "data/gt/ego_poses_val.json")
    args = p.parse_args()

    scenes = list(json.loads(args.splits.read_text())[args.split]) + args.extra_scenes
    t = NuScenesTables(args.dataroot, args.version, scene_names=scenes, keyframes_only=True)
    logs = {r["token"]: r["location"] for r in iter_json_array(args.dataroot / args.version / "log.json")}
    scene_of = {tok: r for tok, r in t.scene.items()}
    out = {}
    for sd in t.sample_data.values():
        if "/LIDAR_TOP/" not in sd["filename"]:
            continue
        sample = t.sample[sd["sample_token"]]
        scene = scene_of[sample["scene_token"]]
        pose = t.ego_pose[sd["ego_pose_token"]]
        out[sd["sample_token"]] = {"scene": scene["name"], "location": logs[scene["log_token"]],
                                   "translation": pose["translation"], "rotation": pose["rotation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"{len(out)} samples from {len(scenes)} scenes -> {args.output}")


if __name__ == "__main__":
    main()
