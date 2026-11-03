"""Datasets over packed frames + cached teacher outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from qdistill.np_io import load_npz
from qdistill.tap_store import decode, load_W

IMAGE_SIZE = (896, 512)  # (W, H), the teacher's input resolution; the head's lidar2img assumes it
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_tokens(splits_file: str | Path, split: str) -> list[str]:
    """Frame tokens of a split. ``split`` may be ``name`` or ``name:N`` (first N entries, e.g. ranked scenes).

    Mini splits files map names to frame tokens. Trainval files (scripts/dataset/select_scenes.py) map names to scene
    names, expanded through ``scenes[name]["samples"]`` (or ``frames[name]`` when frames were subselected);
    names under ``frame_lists`` are explicit frame-token lists (e.g. ablation arms).
    """
    name, _, n = split.partition(":")
    data = json.loads(Path(splits_file).read_text())
    if name in data.get("frame_lists", {}):
        return data["frame_lists"][name][: int(n)] if n else data["frame_lists"][name]
    entries = data[name][: int(n)] if n else data[name]
    if "scenes" not in data:
        return entries
    frames = data.get("frames", {})
    return [t for scene in entries for t in frames.get(scene, data["scenes"][scene]["samples"])]


def load_images(frame_dir: Path, jitter: ColorJitter | None = None) -> torch.Tensor:
    """[N_cam, 3, 512, 896], resized like the teacher (bicubic) and ImageNet-normalised."""
    frame = json.loads((frame_dir / "frame.json").read_text())
    images = []
    for cam in frame["cam_order"]:
        img = Image.open(frame_dir / "images" / f"{cam}.jpg").convert("RGB")
        img = TF.resize(img, [IMAGE_SIZE[1], IMAGE_SIZE[0]], interpolation=InterpolationMode.BICUBIC)
        if jitter is not None:
            img = jitter(img)
        images.append(TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD))
    return torch.stack(images)


def load_resized(resized_root: Path, token: str, jitter: ColorJitter | None = None) -> torch.Tensor:
    """Same output as ``load_images`` from pre-resized uint8 arrays."""
    x = torch.from_numpy(np.load(resized_root / f"{token}.npy")).permute(0, 3, 1, 2)  # [N, 3, H, W] uint8
    x = x.float().div_(255.0)
    if jitter is not None:
        x = torch.stack([jitter(img) for img in x])
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_images_gpu(frame_dir: Path, return_raw: bool = False):
    """``load_images`` on the GPU: nvjpeg decode, bicubic antialias resize rounded to uint8 (as PIL), ImageNet
    normalisation. Not bit-identical to the PIL path (decoder IDCT / chroma and resize kernel: ~0.5 grey levels mean).
    The decoder runs on its own CUDA stream; synchronising on both sides keeps the caching allocator from handing its
    buffers to work still pending on the default stream.
    ``return_raw``: also the decoded full-resolution images [N_cam, 3, 900, 1600] as float 0-255 (the resize input,
    no extra copy; heading refiner crops)."""
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    frame = json.loads((frame_dir / "frame.json").read_text())
    datas = [read_file(str(frame_dir / "images" / f"{cam}.jpg")) for cam in frame["cam_order"]]
    torch.cuda.synchronize()
    imgs = decode_jpeg(datas, mode=ImageReadMode.RGB, device="cuda")
    torch.cuda.synchronize()
    raw = torch.stack(imgs).float()
    x = torch.nn.functional.interpolate(raw, size=(IMAGE_SIZE[1], IMAGE_SIZE[0]), mode="bicubic", antialias=True,
                                        align_corners=False)
    x = x.round_().clamp_(0, 255).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    return ((x - mean) / std, raw) if return_raw else (x - mean) / std


IMAGE_FORMATS = ("resized", "jpeg", "jpeg_gpu")


def load_frame_images(image_root: Path, token: str, image_format: str, jitter: ColorJitter | None = None) -> torch.Tensor:
    """[N_cam, 3, 512, 896] for one frame.

    ``resized``: ``image_root`` = pre-resized arrays, ~17 ms/frame, 7.9 MB/frame on disk.
    ``jpeg``: ``image_root`` = packed frames; decodes the original nuScenes JPEGs and resizes, ~145 ms/frame, no extra disk.
    Both give the teacher's preprocessing (bicubic to 896x512).
    """
    if image_format == "resized":
        return load_resized(image_root, token, jitter)
    if image_format == "jpeg":
        return load_images(image_root / token, jitter)
    if image_format == "jpeg_gpu":  # eval / latency only (no jitter); returns a CUDA tensor
        if jitter is not None:
            raise ValueError("jpeg_gpu has no colour jitter")
        return load_images_gpu(image_root / token)
    raise ValueError(f"unknown image format {image_format!r}, expected one of {IMAGE_FORMATS}")


class TapDistillDataset(Dataset):
    """(images, standardised teacher taps) per frame. Images: pre-resized arrays or packed JPEGs (``image_format``)."""

    def __init__(self, tokens, image_root, cache_root, stats: dict, color_jitter: bool = False, image_format: str = "resized",
                 fg_root: str | Path | None = None, taps_dir: str = "taps", gt_depth_root: str | Path | None = None):
        if image_format not in IMAGE_FORMATS:
            raise ValueError(f"unknown image format {image_format!r}, expected one of {IMAGE_FORMATS}")
        self.tokens = list(tokens)
        self.image_root = Path(image_root)
        self.image_format = image_format
        self.fg_root = None if fg_root is None else Path(fg_root)  # scripts/backbone/precompute_fg.py maps
        self.gt_depth_root = None if gt_depth_root is None else Path(gt_depth_root)  # precompute_fg.py --gt-depth
        self.cache_root = Path(cache_root)
        self.taps_dir = taps_dir  # "taps" (full fp16) or a converted cache, e.g. "taps_vitW+llmInt8"
        self.W_pinv = None
        self.stats = {k: torch.as_tensor(v).float() for k, v in stats.items()}
        self.jitter = ColorJitter(0.2, 0.2, 0.2, 0.02) if color_jitter else None

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> dict:
        token = self.tokens[i]
        taps = load_npz(self.cache_root / self.taps_dir / f"{token}.npz")
        if "vitW" in taps.files and self.W_pinv is None:
            self.W_pinv = load_W(self.cache_root)[1]
        decoded = dict(zip(("vit", "llm"), decode(taps, self.W_pinv)))
        out = {"token": token, "images": load_frame_images(self.image_root, token, self.image_format, self.jitter)}
        if self.fg_root is not None:
            out["fg"] = torch.from_numpy(np.load(self.fg_root / f"{token}.npy")).float()
        if self.gt_depth_root is not None:
            out["gt_depth"] = torch.from_numpy(np.load(self.gt_depth_root / f"{token}.npy")).float()
        for name in ("vit", "llm"):
            x = decoded[name]
            out[name] = ((x - self.stats[f"{name}_mean"]) / self.stats[f"{name}_std"]).to(torch.float16)
        return out
