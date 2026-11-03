"""DINOv2 embeddings of the candidate training frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from qdistill.data import split_tokens
from qdistill.nuscenes_pack import CAM_ORDER

ROOT = Path(__file__).resolve().parents[2]
SIZE = (224, 392)  # (H, W), multiples of the 14-px patch


class FrameImages(Dataset):
    def __init__(self, tokens, packed: Path, mean, std):
        self.tokens, self.packed, self.mean, self.std = tokens, packed, mean, std

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        d = self.packed / self.tokens[i] / "images"
        x = torch.stack([TF.to_tensor(Image.open(d / f"{c}.jpg").convert("RGB").resize(SIZE[::-1], Image.BICUBIC))
                         for c in CAM_ORDER])
        return (x - self.mean) / self.std


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--split", default="train_ranked")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--batch-frames", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)  # RAM:
    args = p.parse_args()

    tokens = split_tokens(args.splits, args.split)
    model = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=SIZE).cuda().eval()
    cfg = model.pretrained_cfg
    mean, std = torch.tensor(cfg["mean"]).view(1, 3, 1, 1), torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    loader = DataLoader(FrameImages(tokens, args.packed, mean, std), batch_size=args.batch_frames,
                        num_workers=args.workers)
    out = np.zeros((len(tokens), len(CAM_ORDER) * model.num_features), dtype=np.float16)
    t0, i = time.perf_counter(), 0
    with torch.no_grad():
        for x in loader:
            b = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x.flatten(0, 1).cuda(non_blocking=True)).float()
            f = torch.nn.functional.normalize(f, dim=1).view(b, -1) / len(CAM_ORDER) ** 0.5
            out[i : i + b] = f.cpu().numpy().astype(np.float16)
            i += b
            if i % 800 < b:
                print(f"  {i}/{len(tokens)} frames, {(time.perf_counter() - t0) / i * 1000:.0f} ms/frame", flush=True)
    np.savez(args.output, tokens=np.array(tokens), emb=out, model=args.model, size=np.array(SIZE))
    (args.output.with_suffix(".json")).write_text(json.dumps(
        {"model": args.model, "size": SIZE, "split": args.split, "frames": len(tokens),
         "s_per_frame": (time.perf_counter() - t0) / len(tokens)}, indent=2))
    print(f"wrote {args.output}: {out.shape}, {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
