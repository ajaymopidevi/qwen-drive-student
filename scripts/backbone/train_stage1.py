"""Stage 1: train the student to regress the teacher taps, with losses in the head space."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from qdistill.data import IMAGE_FORMATS, TapDistillDataset, split_tokens
from qdistill.depth_kd import HeadDepth, adaptor_loss, depth_kl, expected_depth, head_space_loss, range_ramp
from qdistill.student import TapNormalizer, TapStudent, tap_loss

ROOT = Path(__file__).resolve().parents[2]


def collate(batch):
    out = {
        "token": [b["token"] for b in batch],
        "images": torch.stack([b["images"] for b in batch]),
        "vit": torch.stack([b["vit"] for b in batch]),
        "llm": torch.stack([b["llm"] for b in batch]),
    }
    for k in ("fg", "gt_depth"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def extra_losses(vit, batch, norm, head_depth, fg_alpha, depth=True, head_space=False, range_beta=0.0, logs=None):
    """per-patch ViT weights (1 + alpha * fg class map), depth KL and head-space ViT terms through the frozen head.

    Returns (vit_w, kl, kl_fg, hs) with None for parts that are off; hs = (mse, cos) on LayerNorm(W x).
    range_beta > 0 multiplies the depth-KL weight of foreground patches by 1 + beta * ramp(teacher expected depth,
    15 -> 30 m). ``logs`` (dict) receives the far-foreground depth KL (fg patches the teacher puts beyond 20 m).
    with ``batch["gt_depth"]`` (GT box camera depth per patch) the ramp and the far log use it instead; the
    teacher's expected depth is ~45 m on every fg patch (most mass in out-of-grid bins), so the ramp was flat.
    """
    vit_w = 1.0 + fg_alpha * batch["fg"].cuda(non_blocking=True) if fg_alpha > 0 else None
    kl = kl_fg = hs = None
    if head_space:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_hs = head_depth.head_space(norm.denormalize("vit", vit))
            with torch.no_grad():
                t_hs = head_depth.head_space(norm.denormalize("vit", batch["vit"].cuda(non_blocking=True)))
        hs = head_space_loss(s_hs, t_hs, vit_w)
    if head_depth is not None and depth:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits = head_depth(norm.denormalize("vit", vit))
            with torch.no_grad():
                t_logits = head_depth(norm.denormalize("vit", batch["vit"].cuda(non_blocking=True)))
        fg = (batch["fg"].cuda() > 0).float() if "fg" in batch else None
        w_depth = vit_w
        if fg is not None and (range_beta > 0 or logs is not None):
            if "gt_depth" in batch:
                d_t = batch["gt_depth"].cuda()
            else:
                d_t = expected_depth(t_logits, head_depth.depth_bins)
            if range_beta > 0:
                w_depth = (vit_w if vit_w is not None else 1.0) * (1.0 + range_beta * fg * range_ramp(d_t))
            if logs is not None:
                far = fg * (d_t > 20.0).float()
                if far.sum() > 0:
                    logs["depth_kl_far"] = depth_kl(s_logits, t_logits, far).detach()
        kl = depth_kl(s_logits, t_logits, w_depth)
        if fg is not None:
            kl_fg = depth_kl(s_logits, t_logits, fg).detach()  # foreground-only, logged
    return vit_w, kl, kl_fg, hs


@torch.no_grad()
def evaluate(model, loader, norm, head_depth=None, head_space=False, adapt=False) -> dict:
    """``loss`` = checkpoint-selection metric: the unweighted tap loss, or with head_space the unweighted head-space
    ViT terms + LLM terms (the raw ViT tap is unconstrained there). Raw tap terms and depth KL are logged."""
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vit, llm = model(batch["images"].cuda(non_blocking=True))
        out = tap_loss({"vit": vit, "llm": llm}, {k: batch[k].cuda() for k in ("vit", "llm")}, norm)
        if head_depth is not None:
            logs = {}
            _, kl, kl_fg, hs = extra_losses(vit, batch, norm, head_depth, 0.0, depth=True, head_space=head_space, logs=logs)
            out.update(logs)
            if adapt:
                out["adapt_mse"], out["adapt_cos"] = adaptor_loss(head_depth.adaptor, norm.denormalize("llm", llm),
                                                                  norm.denormalize("llm", batch["llm"].cuda()))
            out["depth_kl"] = kl
            if kl_fg is not None:
                out["depth_kl_fg"] = kl_fg
            if hs is not None:
                out["hs_vit_mse"], out["hs_vit_cos"] = hs
                out["raw_loss"] = out["loss"]
                out["loss"] = hs[0] + (1 - hs[1]) + out["llm_mse"] + (1 - out["llm_cos"])
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(batch["token"])
        n += len(batch["token"])
    model.train()
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--backbone", default="hgnetv2_b4.ssld_stage2_ft_in1k")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cos-weight", type=float, default=1.0)
    p.add_argument("--color-jitter", action="store_true")
    p.add_argument("--workers", type=int, default=1)  # 4 workers exhausted WSL RAM
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-mini")
    p.add_argument("--resized", type=Path, default=ROOT / "data/resized/v1.0-mini")
    p.add_argument("--images", choices=IMAGE_FORMATS, default="resized")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-mini")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-mini.json")
    p.add_argument("--train-split", default="mini_train")
    p.add_argument("--val-split", default="mini_val")
    p.add_argument("--tap-stats", type=Path, default=None, help="default: <cache>/tap_stats_<train-split>.npz")
    p.add_argument("--fg-root", type=Path, default=None, help="precomputed fg maps (scripts/backbone/precompute_fg.py), ")
    p.add_argument("--fg-alpha", type=float, default=0.0, help="ViT tap loss weight per patch = 1 + alpha * fg class map")
    p.add_argument("--depth-kd", type=float, default=0.0, help="weight of the depth KL through the frozen head")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--head-space", type=float, default=0.0,
                   help="weight of the head-space ViT loss on LayerNorm(W x); also selects checkpoints on it")
    p.add_argument("--vit-raw-weight", type=float, default=1.0, help="weight of the raw ViT tap terms (0 = head-space only)")
    p.add_argument("--taps-dir", default="taps",
                   help="tap subdirectory of --cache for the train split: taps (full) or a converted cache (tap_store.py)")
    p.add_argument("--depth-range-beta", type=float, default=0.0,
                   help="depth-KL weight on fg patches x (1 + beta * ramp(teacher depth, 15 -> 30 m))")
    p.add_argument("--llm-adapt-weight", type=float, default=0.0,
                   help="weight of the LLM loss through the frozen adaptor (MSE/var + 1 - cos over its 4 scales)")
    p.add_argument("--gt-depth-root", type=Path, default=None,
                   help="GT depth maps (precompute_fg.py --gt-depth): range weight / far log from GT depth")
    p.add_argument("--val-taps-dir", default="taps", help="same for the val split")
    p.add_argument("--extra-train", action="append", default=[], metavar="SPLIT,IMAGES,CACHE",
                   help="extra training split with its own image root (see --images) and teacher-cache root (repeatable)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = dict(np.load(args.tap_stats or args.cache / f"tap_stats_{args.train_split.replace(':', '_')}.npz"))

    image_root = args.packed if args.images == "jpeg" else args.resized
    train_parts = [TapDistillDataset(split_tokens(args.splits, args.train_split), image_root, args.cache, stats,
                                     args.color_jitter, args.images, fg_root=args.fg_root, taps_dir=args.taps_dir,
                                     gt_depth_root=args.gt_depth_root)]
    for spec in args.extra_train:
        split, images, cache = spec.split(",")
        train_parts.append(TapDistillDataset(split_tokens(args.splits, split), ROOT / images, ROOT / cache, stats,
                                             args.color_jitter, args.images))
    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_ds = TapDistillDataset(split_tokens(args.splits, args.val_split), image_root, args.cache, stats, image_format=args.images,
                               fg_root=args.fg_root, taps_dir=args.val_taps_dir, gt_depth_root=args.gt_depth_root)
    if args.fg_alpha > 0 and args.fg_root is None:
        raise SystemExit("--fg-alpha needs --fg-root")
    head_depth = HeadDepth(args.weights) if (args.depth_kd > 0 or args.head_space > 0 or args.llm_adapt_weight > 0) else None
    if args.depth_range_beta > 0 and (args.depth_kd <= 0 or args.fg_root is None):
        raise SystemExit("--depth-range-beta needs --depth-kd > 0 and --fg-root")
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    # Validation loads in the main process: a second forked worker pushed WSL RAM to ~1 GB free.
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=0, collate_fn=collate)

    model = TapStudent(args.backbone).cuda()
    model.set_grad_checkpointing(True)
    norm = TapNormalizer(stats).cuda()
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(q) for q in backbone_params}
    other_params = [q for q in model.parameters() if id(q) not in backbone_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * args.backbone_lr_mult}, {"params": other_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_dl)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step: int) -> None:
        if step < args.warmup_steps:
            f = (step + 1) / args.warmup_steps
        else:
            f = 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    n_params = sum(q.numel() for q in model.parameters())
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'runs/mlflow.db'}")
    mlflow.set_experiment("stage1_tap_distill")
    step, best, history, start = 0, float("inf"), [], 1
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({**{k: str(v) for k, v in vars(args).items()}, "student_params_M": round(n_params / 1e6, 2),
                           "train_frames": len(train_ds), "val_frames": len(val_ds),
                           "train_frames_per_part": ",".join(str(len(d)) for d in train_parts)})
        t0 = time.perf_counter() - (history[-1]["elapsed_min"] * 60 if history else 0)
        for epoch in range(start, args.epochs + 1):
            sums, n = {}, 0
            for batch in train_dl:
                set_lr(step)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vit, llm = model(batch["images"].cuda(non_blocking=True))
                vit_w, kl, kl_fg, hs = extra_losses(vit, batch, norm, head_depth, args.fg_alpha,
                                                    depth=args.depth_kd > 0, head_space=args.head_space > 0,
                                                    range_beta=args.depth_range_beta)
                out = tap_loss({"vit": vit, "llm": llm},
                               {k: batch[k].cuda(non_blocking=True) for k in ("vit", "llm")}, norm, args.cos_weight,
                               vit_weight=vit_w, vit_scale=args.vit_raw_weight)
                if hs is not None:
                    out["loss"] = out["loss"] + args.head_space * (hs[0] + (1 - hs[1]))
                    out["hs_vit_mse"], out["hs_vit_cos"] = hs[0].detach(), hs[1].detach()
                if args.llm_adapt_weight > 0:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        a_mse, a_cos = adaptor_loss(head_depth.adaptor, norm.denormalize("llm", llm),
                                                    norm.denormalize("llm", batch["llm"].cuda(non_blocking=True)))
                    out["loss"] = out["loss"] + args.llm_adapt_weight * (a_mse + (1 - a_cos))
                    out["adapt_mse"], out["adapt_cos"] = a_mse.detach(), a_cos.detach()
                if kl is not None:
                    out["loss"] = out["loss"] + args.depth_kd * kl
                    out["depth_kl"] = kl.detach()
                    if kl_fg is not None:
                        out["depth_kl_fg"] = kl_fg
                out["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                for k, v in out.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                n += 1
                step += 1
                if step % 20 == 0:
                    mlflow.log_metrics({f"train/{k}": float(v) for k, v in out.items()} |
                                       {"train/grad_norm": float(grad_norm), "lr": opt.param_groups[1]["lr"]}, step=step)
            rec = {"epoch": epoch, "step": step, "elapsed_min": (time.perf_counter() - t0) / 60,
                   **{f"train_{k}": v / n for k, v in sums.items()}}
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val = evaluate(model, val_dl, norm, head_depth, head_space=args.head_space > 0,
                               adapt=args.llm_adapt_weight > 0)
                rec.update({f"val_{k}": v for k, v in val.items()})
                mlflow.log_metrics({f"val/{k}": v for k, v in val.items()}, step=step)
                if val["loss"] < best:
                    best = val["loss"]
                    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, run_dir / "best.pt")
            history.append(rec)
            tmp = run_dir / "last.pt.tmp"
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "opt": opt.state_dict(),
                        "step": step, "best": best, "history": history, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state()}, tmp)
            tmp.replace(run_dir / "last.pt")
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
        mlflow.log_metric("best_val_loss", best)
        mlflow.log_artifact(str(run_dir / "history.json"))
        print(f"done: best val loss {best:.4f}, {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
