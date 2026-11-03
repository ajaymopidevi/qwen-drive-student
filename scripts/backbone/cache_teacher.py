"""Run the teacher on packed frames and cache its taps and head outputs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from qdistill import tap_store
from qdistill.data import split_tokens
from qdistill.teacher import PRECISIONS, extract_taps, load_head, load_processor, load_vlm, run_head
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=ROOT / "data/packed/v1.0-mini")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/teacher/v1.0-mini")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    parser.add_argument("--precision", choices=PRECISIONS, default="bf16")
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", type=Path, default=None, help="restrict to --split frames (see data.split_tokens)")
    parser.add_argument("--split", default=None, help="split name, or name:N for the first N entries")
    parser.add_argument("--drop-occ-logits", action="store_true",
                        help="do not store occ_logits (12.8 of ~50 MB/frame); decoded occ is kept and the logits can be "
                             "regenerated from the taps with --phase head")
    parser.add_argument("--phase", choices=("all", "vlm", "head"), default="all")
    parser.add_argument("--tap-format", choices=tap_store.FORMATS, default="full")
    parser.add_argument("--chunk", type=int, default=None, help="frames per VLM -> head round (phase all only)")
    parser.add_argument("--discard-taps", action="store_true",
                        help="delete taps this run wrote once the frame's head output is saved (needs --phase all)")
    args = parser.parse_args()
    if (args.chunk or args.discard_taps) and args.phase != "all":
        raise SystemExit("--chunk / --discard-taps need --phase all")

    full_dir, head_dir = args.output / "taps", args.output / "head"
    taps_dir = full_dir if args.tap_format == "full" else args.output / f"taps_{args.tap_format}"
    W = W_pinv = None
    if args.tap_format != "full":
        W, W_pinv = tap_store.load_W(args.output)

    def save_taps(name: str, vit: torch.Tensor, llm: torch.Tensor) -> None:
        tmp = taps_dir / f"{name}.tmp.npz"
        if args.tap_format == "full":
            np.savez(tmp, vit=vit.to(torch.float16).cpu().numpy(), llm=llm.to(torch.float16).cpu().numpy())
        else:
            np.savez(tmp, **tap_store.encode(args.tap_format, vit.cpu(), llm.cpu(), W))
        tmp.rename(taps_dir / f"{name}.npz")

    taps_dir.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True)
    if args.splits is not None:
        frame_dirs = [args.frames / t for t in split_tokens(args.splits, args.split)][: args.limit]
        missing = [d for d in frame_dirs if not (d / "frame.json").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} frames of {args.split} are not packed, e.g. {missing[0]}")
    else:
        frame_dirs = sorted(p for p in args.frames.iterdir() if p.is_dir())[: args.limit]
    processor = load_processor(args.weights)
    stats = {"precision": args.precision, "gpu_layers": args.gpu_layers, "frames": len(frame_dirs)}
    chunks = [frame_dirs[i : i + args.chunk] for i in range(0, len(frame_dirs), args.chunk)] if args.chunk else [frame_dirs]
    for k, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"chunk {k + 1}/{len(chunks)}: {len(chunk)} frames", flush=True)
        run_chunk(args, chunk, processor, taps_dir, full_dir, head_dir, save_taps, W_pinv, stats)

    (args.output / f"stats_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def run_chunk(args, frame_dirs, processor, taps_dir, full_dir, head_dir, save_taps, W_pinv, stats) -> None:
    written = []  # taps this run created (only these are discarded)
    if args.phase in ("all", "vlm"):
        todo = [d for d in frame_dirs if not (taps_dir / f"{d.name}.npz").exists()]
        if taps_dir != full_dir:
            have_full = [d for d in todo if (full_dir / f"{d.name}.npz").exists()]
            for d in tqdm(have_full, desc="convert"):
                with np.load(full_dir / f"{d.name}.npz") as z:
                    save_taps(d.name, torch.from_numpy(z["vit"]).float(), torch.from_numpy(z["llm"]).float())
                written.append(d.name)
            stats["converted_from_full"] = len(have_full)
            todo = [d for d in todo if d not in have_full]
        if todo:
            vlm = load_vlm(args.weights, args.precision, gpu_layers=args.gpu_layers)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for d in tqdm(todo, desc="vlm"):
                frame = PerceptionFrame(d)
                inputs, _ = processor(frame, device="cuda")
                vit, llm = extract_taps(vlm, inputs, num_cams=len(frame.cam_order))
                # fp16 first, as the full cache stores it, so a compressed file equals converting the full one
                save_taps(d.name, vit.to(torch.float16).float(), llm.to(torch.float16).float())
                written.append(d.name)
            stats.setdefault("vlm_s_per_frame", []).append((time.perf_counter() - t0) / len(todo))
            stats["vlm_gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
            del vlm
            gc.collect()
            torch.cuda.empty_cache()

    if args.phase in ("all", "head"):
        todo = [d for d in frame_dirs if not (head_dir / f"{d.name}.npz").exists()]
        if todo:
            head = load_head(args.weights)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for d in tqdm(todo, desc="head"):
                frame = PerceptionFrame(d)
                _, img_metas = processor(frame)
                vit, llm = tap_store.decode(np.load(taps_dir / f"{d.name}.npz"), W_pinv)
                result = run_head(head, vit, llm, img_metas)
                if args.drop_occ_logits:
                    result.pop("occ_logits")
                tmp = head_dir / f"{d.name}.tmp.npz"
                np.savez(tmp, **result)
                tmp.rename(head_dir / f"{d.name}.npz")
            stats.setdefault("head_s_per_frame", []).append((time.perf_counter() - t0) / len(todo))
            stats["head_gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
            del head
            gc.collect()
            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
