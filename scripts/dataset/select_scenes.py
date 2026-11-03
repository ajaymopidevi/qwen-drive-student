"""Rank nuScenes scenes for selection."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from qdistill.json_stream import iter_json_array
from qdistill.nuscenes_pack import CAM_ORDER, LABEL_MAP
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]
RARE = ("bicycle", "generic_object", "traffic_cone", "barrier")
EMBED_SIZE = (256, 448)  # (H, W) for the selection embedding only; aspect of the teacher's 512x896


def load_metadata(table_dir: Path) -> tuple[dict, dict]:
    """Per-scene info and keyframe camera filenames per sample, streaming the big tables."""
    logs = {r["token"]: r["location"] for r in json.loads((table_dir / "log.json").read_text())}
    cats = {r["token"]: r["name"] for r in json.loads((table_dir / "category.json").read_text())}
    inst_cat = {r["token"]: cats[r["category_token"]] for r in iter_json_array(table_dir / "instance.json")}
    samples = json.loads((table_dir / "sample.json").read_text())
    scene_of = {s["token"]: s["scene_token"] for s in samples}

    counts = defaultdict(Counter)
    for a in iter_json_array(table_dir / "sample_annotation.json"):
        label = LABEL_MAP.get(inst_cat[a["instance_token"]], -1)
        if label >= 0 and a["num_lidar_pts"] + a["num_radar_pts"] > 0:
            counts[scene_of[a["sample_token"]]][DET_CLASS_NAMES[label]] += 1

    cam_files = defaultdict(dict)
    for sd in iter_json_array(table_dir / "sample_data.json"):
        if sd["is_key_frame"] and sd["filename"].startswith("samples/CAM_"):
            cam_files[sd["sample_token"]][sd["filename"].split("/")[1]] = sd["filename"]

    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s["scene_token"]].append((s["timestamp"], s["token"]))
    scenes = {}
    for sc in json.loads((table_dir / "scene.json").read_text()):
        desc = sc["description"].lower()
        night, rain = bool(re.search(r"\bnight\b", desc)), bool(re.search(r"\brain\b", desc))
        scenes[sc["name"]] = {
            "group": f"{logs[sc['log_token']]}|{'night' if night else 'day'}|{'rain' if rain else 'dry'}",
            "frames": sc["nbr_samples"],
            "counts": {c: counts[sc["token"]][c] for c in DET_CLASS_NAMES if counts[sc["token"]][c]},
            "samples": [t for _, t in sorted(by_scene[sc["token"]])],
        }
    return scenes, cam_files


@torch.no_grad()
def embed_scenes(names, scenes, cam_files, dataroot: Path, n_frames: int, backbone: str) -> np.ndarray:
    model = timm.create_model(backbone, pretrained=True, num_classes=0).cuda().eval()
    cfg = model.pretrained_cfg
    mean, std = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).cuda(), torch.tensor(cfg["std"]).view(1, 3, 1, 1).cuda()
    out = np.zeros((len(names), model.num_features), dtype=np.float32)
    for i, name in enumerate(names):
        toks = scenes[name]["samples"]
        picks = [toks[j] for j in np.linspace(0, len(toks) - 1, n_frames).round().astype(int)]
        imgs = [TF.to_tensor(Image.open(dataroot / cam_files[t][cam]).convert("RGB").resize(EMBED_SIZE[::-1], Image.BILINEAR))
                for t in picks for cam in CAM_ORDER]
        x = (torch.stack(imgs).cuda() - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model(x).float()
        f = torch.nn.functional.normalize(f, dim=1).mean(0)
        out[i] = torch.nn.functional.normalize(f, dim=0).cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"  embedded {i + 1}/{len(names)}", flush=True)
    return out


def rank(names, scenes, emb: np.ndarray, seeds, lam: float) -> list[str]:
    """Proportional interleaving over groups + greedy rare-class gain / core-set distance within a group."""
    idx = {n: i for i, n in enumerate(names)}
    groups = defaultdict(list)
    for n in names:
        groups[scenes[n]["group"]].append(n)
    share = {g: len(v) / len(names) for g, v in groups.items()}
    order = [n for n in seeds if n in idx]
    remaining = {g: [n for n in v if n not in order] for g, v in groups.items()}
    taken = Counter(scenes[n]["group"] for n in order)
    acc = Counter()
    for n in order:
        acc.update(scenes[n]["counts"])
    min_dist = np.full(len(names), 2.0)  # cosine distance <= 2
    for n in order:
        min_dist = np.minimum(min_dist, 1.0 - emb @ emb[idx[n]])

    while any(remaining.values()):
        k = len(order) + 1
        open_groups = [g for g in groups if remaining[g]]
        # Largest deficit vs the proportional target; groups not yet reached get a floor of one.
        g = max(open_groups, key=lambda g: (max(share[g] * k, 1.0 if taken[g] == 0 else 0.0) - taken[g], share[g]))
        cand = remaining[g]
        gain = np.array([sum(math.log1p(scenes[n]["counts"].get(c, 0) / (1 + acc[c])) for c in RARE) for n in cand])
        dist = np.array([min_dist[idx[n]] for n in cand])
        score = gain / max(gain.max(), 1e-9) + lam * dist / max(dist.max(), 1e-9)
        pick = cand[int(score.argmax())]
        order.append(pick)
        cand.remove(pick)
        taken[g] += 1
        acc.update(scenes[pick]["counts"])
        min_dist = np.minimum(min_dist, 1.0 - emb @ emb[idx[pick]])
    return order


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--dataroot", type=Path, default=ROOT / "data/nuscenes")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--official", type=Path, default=ROOT / "data/splits/nuscenes_official_scenes.json")
    p.add_argument("--backbone", default="hgnetv2_b4.ssld_stage2_ft_in1k")
    p.add_argument("--embed-frames", type=int, default=5)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the core-set distance vs the rare-class gain")
    p.add_argument("--val-quick", type=int, default=30)
    args = p.parse_args()

    official = json.loads(args.official.read_text())
    scenes, cam_files = load_metadata(args.dataroot / args.version)
    print(f"{len(scenes)} scenes in metadata")

    missing = [n for n in official["train"] + official["val"]
               if any(not (args.dataroot / cam_files[t][c]).exists() for t in scenes[n]["samples"] for c in CAM_ORDER)]
    if missing:
        raise SystemExit(f"{len(missing)} train/val scenes have missing camera keyframes, e.g. {missing[:5]}")

    emb_file = args.output.with_name(args.output.stem + f"_embed_{args.embed_frames}f.npz")
    names_all = official["train"] + official["val"]
    if emb_file.exists() and list(np.load(emb_file)["names"]) == names_all:
        emb_all = np.load(emb_file)["emb"]
        print(f"embeddings from {emb_file}")
    else:
        emb_all = embed_scenes(names_all, scenes, cam_files, args.dataroot, args.embed_frames, args.backbone)
        np.savez(emb_file, names=np.array(names_all), emb=emb_all)
    n_train = len(official["train"])

    train = rank(official["train"], scenes, emb_all[:n_train], official["mini_train"], args.lam)
    val = rank(official["val"], scenes, emb_all[n_train:], official["mini_val"], args.lam)
    out = {
        "source": "scripts/dataset/select_scenes.py",
        "params": {"backbone": args.backbone, "embed_frames": args.embed_frames, "lam": args.lam, "rare": RARE},
        "train_ranked": train,
        "val": val,
        "val_quick": val[: args.val_quick],
        "scenes": {n: scenes[n] for n in names_all},
    }
    args.output.write_text(json.dumps(out, indent=1))

    def mix(lst):
        return Counter(scenes[n]["group"] for n in lst)
    print(f"train {len(train)} scenes, val {len(val)}; wrote {args.output}")
    for k in (20, 50, 100, 200, len(train)):
        rare = Counter()
        for n in train[:k]:
            rare.update(scenes[n]["counts"])
        print(f"  first {k:3d} train scenes: {sum(scenes[n]['frames'] for n in train[:k]):5d} frames, rare boxes "
              + ", ".join(f"{c} {rare[c]}" for c in RARE) + f"; groups {dict(mix(train[:k]))}")
    print(f"  val_quick groups {dict(mix(val[: args.val_quick]))}")


if __name__ == "__main__":
    main()
