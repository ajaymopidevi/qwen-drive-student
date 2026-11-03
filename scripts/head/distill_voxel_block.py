"""Distill the voxel block: feature KD on matched inputs + prediction KD through the frozen head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from qdistill.data import IMAGE_SIZE, load_images, split_tokens
from qdistill.light_blocks import (LightVoxelEncoder, TeacherShapedVoxelEncoder, init_from_teacher, pooled_volume,
                                   teacher_encoding, teacher_preact_stats)
from qdistill.student import TapNormalizer, TapStudent
from qdistill.teacher import load_head
from qdistill.voxel_fg import bev_fg_mask
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]


class Frames(Dataset):
    """(images, img_metas, fg) per frame; fg = the [Y, X] BEV foreground mask when ``grid`` is given, or with
    ``class_weights`` a float32 [Y, X] weight map: background 1, a cell inside boxes
    of several classes takes the largest weight."""

    def __init__(self, tokens, packed: Path, grid=None, class_weights=None):
        self.tokens, self.packed, self.grid, self.class_weights = list(tokens), packed, grid, class_weights

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        fd = self.packed / self.tokens[i]
        frame = PerceptionFrame(fd)
        fg = bev_fg_mask(frame.gt, *self.grid) if self.grid is not None else None
        if self.class_weights:
            w = np.ones(fg.shape, dtype=np.float32)
            for k, wk in self.class_weights.items():
                np.maximum(w, np.where(bev_fg_mask(frame.gt, *self.grid, labels={k}), wk, 1.0), out=w)
            fg = w
        return load_images(fd), frame.img_metas(image_size=IMAGE_SIZE), fg


def _as_is(item):
    """Keep img_metas as numpy (default_convert would make CPU tensors, which upstream then treats as tensors)."""
    return item


def vit_path(bev, vit):
    """Upstream forward's ViT branch up to view_trans: neck + DepthNet. ``vit`` [N_cam, 32, 56, 1024]."""
    n = vit.shape[0]
    feats = bev.vit_neck(vit.permute(0, 3, 1, 2))
    mlvl = [f.view(1, n, *f.shape[1:]) for f in feats]
    depths = [bev.depth_net(f.view(-1, *f.shape[-3:])).softmax(dim=1) for f in mlvl]
    return mlvl, depths


def nmse(a, b):
    a, b = a.float(), b.float()
    return ((a - b) ** 2).mean() / b.var().clamp_min(1e-12)


def step_tensors(student, norm, head, images, metas, layers=False):
    """No-grad part: pooled volume s and the teacher block's output t (with ``layers``: every layer's output, the
    last being t)."""
    bev = head.bev_modeling
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vit_n, _ = student(images[None].cuda(non_blocking=True))
        vit = norm.denormalize("vit", vit_n[0]).to(torch.bfloat16)
        mlvl, depths = vit_path(bev, vit)
        s = pooled_volume(bev.view_trans, mlvl, depths, [copy.deepcopy(metas)])
        if layers:  # intermediates kept only on the loss's (Y, X) subsample, as produced
            x = s.permute(0, 1, 5, 4, 3, 2).contiguous().sum(1)
            t = []
            for k, layer in enumerate(bev.view_trans.conv_layer):
                x = layer(x)
                if k < len(bev.view_trans.conv_layer) - 1:
                    t.append(x[..., ::LAYER_STRIDE, ::LAYER_STRIDE].contiguous())
            t.append(x)
        else:
            t = teacher_encoding(bev.view_trans, s)
    return s, t


def step_tensors_llm(student, norm, head, images, metas):
    """step_tensors plus the LLM tap's adaptor features (the head's other input), for prediction-level KD.
    One student forward; all no-grad."""
    from qdistill.pred_kd import llm_mlvl

    bev = head.bev_modeling
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vit_n, llm_n = student(images[None].cuda(non_blocking=True))
        vit = norm.denormalize("vit", vit_n[0]).to(torch.bfloat16)
        llm = norm.denormalize("llm", llm_n[0]).to(torch.bfloat16)
        mlvl, depths = vit_path(bev, vit)
        s = pooled_volume(bev.view_trans, mlvl, depths, [copy.deepcopy(metas)])
        t = teacher_encoding(bev.view_trans, s)
        mlvl_llm = llm_mlvl(bev, llm)
    return s, t, mlvl_llm


LAYER_STRIDE = 4  # 1/16 of the voxels (~40k x 256 values per layer); 2 did not fit at 8 GB


def layer_losses(ys, ts, weight=0.0):
    """nMSE of each block layer's 256-ch output vs the teacher's same layer, all but the last (covered by
    "raw"), on every LAYER_STRIDE-th voxel in X and Y. Block layers are [B, C, X, Y, Z]; the
    teacher's arrive subsampled from step_tensors as [B, C, Z, Y', X'].
    The values are computed without autograd. When the block is training and ``weight`` > 0, the gradient
    weight * d(nMSE)/dy is added by a hook on each layer output instead: autograd through the strided slice built
    extra full-volume gradient buffers and OOMed at 8 GB."""
    out = {}
    for i, (y, t) in enumerate(zip(ys[:-1], ts[:-1])):
        t = t.permute(0, 1, 4, 3, 2)
        with torch.no_grad():
            ys_ = y[:, :, ::LAYER_STRIDE, ::LAYER_STRIDE, :]
            var = t.float().var().clamp_min(1e-12)
            out[f"l{i}"] = ((ys_.float() - t.float()) ** 2).mean() / var
        if weight > 0 and y.requires_grad:
            coef = 2.0 * weight / (t.numel() * var)

            # The closure must not hold y itself: y -> hook -> y is a reference cycle that only the GC frees,
            # which leaked two layer outputs (~0.77 GiB) per step. A detached alias breaks it.
            # In place, no clone: the incoming gradient of a layer output is consumed only by
            # that layer's own backward (the next layer's gradient has already been formed).
            def hook(g, yd=y.detach(), t=t, coef=coef):
                sub = g[:, :, ::LAYER_STRIDE, ::LAYER_STRIDE, :]
                sub.add_(yd[:, :, ::LAYER_STRIDE, ::LAYER_STRIDE, :] - t.to(yd.dtype), alpha=float(coef))

            y.register_hook(hook)
    return out


def losses(head, y, t, raw_weight, fg=None, fg_weight=1.0, cells_out=None):
    """bev_q + occ_p + raw_weight x raw, each an MSE normalised by the target's variance. Errors are first averaged
    per BEV cell (over channels and heights), so the plain mean equals the element mean. With ``fg`` ([Y, X] bool,
    ) and ``fg_weight`` != 1 the loss uses the cell-weighted mean (foreground cells x fg_weight); a float ``fg``
    is the per-cell weight map itself. The logged parts stay
    unweighted, plus the foreground / background split of each term. ``cells_out`` (a dict) receives
    the detached per-cell errors [B, Y, X]."""
    bev = head.bev_modeling
    occ_proj = bev.head.transformer.uvtr_occ_proj
    B, C, Z, H, W = t.shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            q_t = bev.uvtr_query_proj(t.reshape(B, C * Z, H, W))
            o_t = occ_proj(t)
        q_s = bev.uvtr_query_proj(y.reshape(B, C * Z, H, W))
        o_s = occ_proj(y)
    cells = {  # [B, Y, X] per-cell squared error, already divided by the target variance
        "bev_q": ((q_s.float() - q_t.float()) ** 2).mean(1) / q_t.float().var().clamp_min(1e-12),
        "occ_p": ((o_s.float() - o_t.float()) ** 2).mean((1, 2)) / o_t.float().var().clamp_min(1e-12),
        "raw": ((y.float() - t.float()) ** 2).mean((1, 2)) / t.float().var().clamp_min(1e-12),
    }
    parts = {k: v.mean() for k, v in cells.items()}
    if cells_out is not None:
        cells_out.update({k: v.detach() for k, v in cells.items()})
        # per-region target variance: frame variance + per-cell target moments [B, Y, X]
        with torch.no_grad():
            for k, (x, dims) in {"bev_q": (q_t, 1), "occ_p": (o_t, (1, 2)), "raw": (t, (1, 2))}.items():
                x = x.float()
                cells_out[f"{k}_var"] = x.var().clamp_min(1e-12)
                cells_out[f"{k}_t1"] = x.mean(dims)
                cells_out[f"{k}_t2"] = (x ** 2).mean(dims)
    if fg is not None and fg.is_floating_point():
        w, fg = fg[None], fg > 1
        terms = {k: (v * w).sum() / w.sum() for k, v in cells.items()}
    elif fg is None or fg_weight == 1.0:
        terms = parts
    else:
        w = 1.0 + (fg_weight - 1.0) * fg.float()[None]
        terms = {k: (v * w).sum() / w.sum() for k, v in cells.items()}
    if fg is not None:
        with torch.no_grad():
            m = fg[None].expand_as(cells["bev_q"])
            for k, v in cells.items():
                parts[f"{k}_fg"] = v[m].mean() if m.any() else v.new_tensor(float("nan"))
                parts[f"{k}_bg"] = v[~m].mean()
    return terms["bev_q"] + terms["occ_p"] + raw_weight * terms["raw"], parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--student", type=Path, default=ROOT / "weights/student.pt")
    p.add_argument("--train-split", default="s1_C4k")
    p.add_argument("--val-split", default="val_quick")
    p.add_argument("--val-stride", type=int, default=12, help="val_quick every n-th frame for the val loss (~100)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--mid", type=int, default=64)
    p.add_argument("--n-mid", type=int, default=2)
    p.add_argument("--dilations", default=None, help="comma list, one per mid conv; default all 1")
    p.add_argument("--arch", choices=["light", "teacher_shaped"], default="light",
                   help="teacher_shaped: upstream's 3 x 3x3x3 stack, optionally rank-factored (--rank)")
    p.add_argument("--rank", default="0", help="teacher_shaped: channel rank per conv, one int or a comma list per "
                   "layer (32,64,96); 0 = full 256")
    p.add_argument("--init", choices=["random", "teacher_svd", "teacher_pca"], default="random",
                   help="teacher_shaped: copy the teacher's convs (BN folded), factored by weight SVD or output PCA")
    p.add_argument("--stats-frames", type=int, default=30, help="teacher_pca: s1_C4k frames for the output statistics")
    p.add_argument("--n-layers", type=int, default=3, help="teacher_shaped: number of layers (upstream has 3)")
    p.add_argument("--teacher-layers", default=None,
                   help="teacher_shaped init: comma list of upstream layers to copy, one per layer (0,2)")
    p.add_argument("--stop-after", type=int, default=0,
                   help="stop after this many epochs in this call (schedule still spans --epochs; --resume continues)")
    p.add_argument("--checkpoint-layers", action="store_true", help="recompute layer activations in backward (memory)")
    p.add_argument("--layer-weight", type=float, default=0.0,
                   help="teacher_shaped: weight of each intermediate layer's nMSE vs the teacher's same layer")
    p.add_argument("--fg-weight", type=float, default=1.0,
                   help="loss weight of BEV cells inside (margin-widened) GT boxes; 1 = unweighted")
    p.add_argument("--fg-margin", type=float, default=1.0, help="box footprint widening per side (m)")
    p.add_argument("--fg-report", action="store_true", help="log foreground / background error split (implied by --fg-weight)")
    p.add_argument("--raw-weight", type=float, default=0.1)
    p.add_argument("--pred-weight", type=float, default=0.0, help="weight of the prediction-level KD terms")
    p.add_argument("--pred-occ", action="store_true", help="include the occupancy prediction (refiner backward)")
    p.add_argument("--ckpt-head", action="store_true", help="activation checkpointing through the frozen head")
    p.add_argument("--pred-query-norm", type=float, default=None,
                   help="divide the det query weights by this constant")
    p.add_argument("--pred-class-weights", default="", help="det class term weights, e.g. barrier=2 (others 1)")
    p.add_argument("--init-block", type=Path, default=None, help="start from this block checkpoint (its cfg)")
    p.add_argument("--zhalf", action="store_true", help="the block at 8 height levels (qdistill.zhalf)")
    p.add_argument("--z8-compress", default=None, help="zhalf.CompressedZ8Block mode (stride6, dw, full, mean)")
    p.add_argument("--warp-grid", type=float, default=None, help="warped-grid fine-zone half width b (m)")
    p.add_argument("--warp-head", type=Path, default=None,
                   help="fine-tune final.pt (encoder / heads / warped queries) for the head on the warped grid")
    p.add_argument("--ring-weight", type=float, default=1.0, help="loss weight of the warped grid's ring cells")
    p.add_argument("--save-every", type=int, default=0, help="mid-epoch checkpoint every n steps (last_step.pt)")
    p.add_argument("--max-steps", type=int, default=0, help="smoke test: stop each epoch after this many steps")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    a = ck["args"]
    stats = a.get("tap_stats") or args.cache / f"tap_stats_{a['train_split'].replace(':', '_')}.npz"
    student = TapStudent(a["backbone"], pretrained=False).cuda().eval()
    student.load_state_dict(ck["model"])
    norm = TapNormalizer(dict(np.load(stats))).cuda()
    head = load_head(args.weights)
    warp = None
    if args.warp_grid is not None:  # the deployed head on the warped grid (eager)
        from qdistill import head_chain, warp_grid

        assert args.warp_head is not None, "--warp-grid needs --warp-head"
        skip = ",".join(f"skip_ca:{i}" for i in range(6))
        head_chain.install(head, f"{skip},enc_ckpt:{args.warp_head},heads_ckpt:{args.warp_head}")
        print("warp grid:", warp_grid.install(head, args.warp_grid, "", args.warp_head), flush=True)
        print(f"warp offsets: {warp_grid.enable_offsets(head)} decoder attentions", flush=True)
        warp = warp_grid.Warp(args.warp_grid)
    for q in list(student.parameters()) + list(head.parameters()):
        q.requires_grad_(False)

    train = split_tokens(args.splits, args.train_split)
    use_pred = args.pred_weight > 0
    if use_pred:
        from qdistill.head_grad import enable_head_grad
        from qdistill.pred_kd import head_from_voxel, pred_losses

        enable_head_grad()
    if args.init_block is not None:
        from qdistill.light_blocks import build_voxel_block

        ib = torch.load(args.init_block, map_location="cpu", weights_only=False)
        block = build_voxel_block({**ib["cfg"], "checkpoint": args.checkpoint_layers})
        block.load_state_dict(ib["model"])
        rank = "init-block"
        print(f"init from block {args.init_block} (epoch {ib.get('epoch')})", flush=True)
    elif args.arch == "light":
        dil = tuple(int(d) for d in args.dilations.split(",")) if args.dilations else None
        block = LightVoxelEncoder(mid=args.mid, n_mid=args.n_mid, dilations=dil)
    else:
        ranks = [int(r) for r in args.rank.split(",")]
        rank = None if ranks == [0] else ranks[0] if len(ranks) == 1 else ranks
        block = TeacherShapedVoxelEncoder(rank=rank, n_layers=args.n_layers, bn=args.init == "random",
                                          checkpoint=args.checkpoint_layers)
    if args.z8_compress and not getattr(block, "cfg", {}).get("z8_compress"):  # compression inside the block
        from qdistill.zhalf import CompressedZ8Block, ZHalfBlock

        inner = block.block if isinstance(block, ZHalfBlock) else block
        block = CompressedZ8Block(inner, args.z8_compress)
        print(f"z8 compress: {args.z8_compress}", flush=True)
    elif args.zhalf and not getattr(block, "cfg", {}).get("zhalf"):  # run it at 8 height levels (pair means in,
        from qdistill.zhalf import ZHalfBlock                          # linear back to 16 out); the loss is unchanged
        block = ZHalfBlock(block)
        print("zhalf: block at 8 height levels", flush=True)
    block = block.cuda().to(memory_format=torch.channels_last_3d)
    if args.init_block is None and args.init != "random" and not (args.resume and (out / "last.pt").exists()):
        vt = head.bev_modeling.view_trans
        stats = None
        if args.init == "teacher_pca":
            def vols():
                for tok in train[:: max(1, len(train) // args.stats_frames)][: args.stats_frames]:
                    fd = args.packed / tok
                    yield step_tensors(student, norm, head, load_images(fd), PerceptionFrame(fd).img_metas(image_size=IMAGE_SIZE))[0]
            stats = teacher_preact_stats(vt, vols())
        tl = tuple(int(x) for x in args.teacher_layers.split(",")) if args.teacher_layers else None
        init_from_teacher(block, vt, args.init.removeprefix("teacher_"), stats, tl)
        print(f"init from teacher ({args.init}, rank {rank or 'full'})", flush=True)
    opt = torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=1e-4)
    val = split_tokens(args.splits, args.val_split)[:: args.val_stride]
    if args.max_steps:
        val = val[:5]
    steps_per_epoch = min(len(train), args.max_steps) if args.max_steps else len(train)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    history, start, best = [], 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        block.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        history, start, best = st["history"], st["epoch"] + 1, st["best"]
        print(f"resumed after epoch {st['epoch']}", flush=True)
    skip = 0
    if args.resume and (out / "last_step.pt").exists():
        st = torch.load(out / "last_step.pt", map_location="cpu", weights_only=False)
        if st["epoch"] >= start:  # newer than last.pt: continue inside that epoch
            block.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
            history, start, best, skip = st["history"], st["epoch"], st["best"], st["step"]
            print(f"resumed epoch {start} at step {skip}", flush=True)
    n_params = sum(q.numel() for q in block.parameters())
    print(f"block {block.cfg}: {n_params / 1e6:.2f}M params; train {len(train)} frames, val {len(val)}", flush=True)

    vt = head.bev_modeling.view_trans
    class_weights = None
    use_fg = args.fg_weight != 1.0 or args.fg_report or class_weights is not None
    assert not (use_fg and warp), "warp grid: no 200 x 200 foreground weights"
    ring = None
    if warp is not None:
        def wmean(x, dim):  # fine cells -> warped cell means along ``dim`` (pairs in the ring, as Warp.avg)
            t0, t1 = warp.t0, warp.t1
            pair = lambda z: z.unflatten(dim, (z.shape[dim] // 2, 2)).float().mean(dim + 1).to(x.dtype)
            return torch.cat([pair(x.narrow(dim, 0, t0)), x.narrow(dim, t0, t1 - t0),
                              pair(x.narrow(dim, t1, x.shape[dim] - t1))], dim)
        k = torch.arange(warp.G, device="cuda")
        inner_ = (k >= warp.kc) & (k < warp.kc + 2 * warp.nf)
        if args.ring_weight != 1.0:
            ring = torch.where(inner_[:, None] & inner_[None, :], 1.0, args.ring_weight).float()  # [Y, X]
        print(f"warp grid {warp.G} x {warp.G}, ring weight {args.ring_weight}", flush=True)
    z8 = getattr(block, "cfg", {}).get("z8_compress")
    if z8:
        from qdistill.zhalf import zup

    def student_out(s, t):
        """The block on its input (warped when --warp-grid) -> (y at 16 levels, its target, the loss weight map)."""
        if warp is not None:
            with torch.no_grad():
                s = wmean(wmean(s, 2), 3)                                     # [B, 1, G, G, Z, C]
                t = wmean(wmean(t, 3), 4)                                     # [B, C, Z, G, G]
        y = block(s)
        return (zup(y) if z8 else y), t
    grid = (vt.pc_range, vt.voxel_size, vt.voxel_shape, args.fg_margin) if use_fg else None
    use_layers = args.layer_weight > 0
    assert not use_layers or args.arch == "teacher_shaped", "--layer-weight needs --arch teacher_shaped"
    assert not (use_layers and use_pred), "--layer-weight and --pred-weight are not combined"
    assert not (use_layers and (args.warp_grid is not None or args.z8_compress)), "--layer-weight: 200 x 200, Z 16 only"
    pred_cw = None
    if args.pred_class_weights:
        pred_cw = torch.ones(len(DET_CLASS_NAMES), device="cuda")
        for kv in args.pred_class_weights.split(","):
            c, w = kv.split("=")
            pred_cw[DET_CLASS_NAMES.index(c)] = float(w)
        print(f"pred class weights {dict(zip(DET_CLASS_NAMES, pred_cw.tolist()))}", flush=True)

    def forward_loss(images, metas, fg=None):
        fg = torch.from_numpy(fg).cuda() if fg is not None else None
        if use_pred:
            s, t, mlvl = step_tensors_llm(student, norm, head, images, metas)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, t = student_out(s, t)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                to = head_from_voxel(head.bev_modeling, mlvl, t, copy.deepcopy(metas), occ=args.pred_occ)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                so = head_from_voxel(head.bev_modeling, mlvl, y, copy.deepcopy(metas), occ=args.pred_occ,
                                     ckpt=args.ckpt_head and torch.is_grad_enabled())
            loss, parts = losses(head, y, t, args.raw_weight, ring if fg is None else fg, args.fg_weight)
            pl = pred_losses(so, to, pred_cw, args.pred_query_norm)
            del so, to, mlvl
            return s, t, y, loss + args.pred_weight * sum(pl.values()), {**parts, **pl}
        s, t = step_tensors(student, norm, head, images, metas, layers=use_layers)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if use_layers:
                y = block(s, return_layers=True)
            else:
                y, t = student_out(s, t)
        if not use_layers:
            loss, parts = losses(head, y, t, args.raw_weight, ring if fg is None else fg, args.fg_weight)
            return s, t, y, loss, parts
        loss, parts = losses(head, y[-1], t[-1], args.raw_weight, fg, args.fg_weight)
        lp = layer_losses(y, t, args.layer_weight)  # gradient via hooks; values added for logging only
        loss = loss + args.layer_weight * sum(lp.values())
        return s, t, y, loss, {**parts, **lp}

    val_ds = Frames(val, args.packed, grid, class_weights)

    def run_val():
        block.eval()
        acc = {}
        with torch.no_grad():
            for k_ in range(len(val_ds)):
                s, t, y, loss, parts = forward_loss(*val_ds[k_])
                acc.setdefault("loss", []).append(float(loss))
                for k, v in parts.items():
                    acc.setdefault(k, []).append(float(v))
                del s, t, y
        block.train()
        torch.cuda.empty_cache()  # val / init passes leave a differently shaped cache
        return {k: float(np.nanmean(v)) for k, v in acc.items()}

    if start == 0 and skip == 0 and (args.init != "random" or args.init_block is not None):
        v0 = run_val()
        history.append({"epoch": -1, "val": v0, "minutes": 0.0})
        print(f"init val {json.dumps(v0)}", flush=True)
    end = min(args.epochs, start + args.stop_after) if args.stop_after else args.epochs
    for epoch in range(start, end):
        g = torch.Generator().manual_seed(args.seed + epoch)
        if args.save_every:  # an explicit seeded order, so a mid-epoch resume continues it
            order = [train[k] for k in torch.randperm(len(train), generator=g).tolist()]
            first = skip if epoch == start else 0
            loader = DataLoader(Frames(order[first:], args.packed, grid, class_weights), batch_size=None, shuffle=False,
                                num_workers=1, collate_fn=_as_is, pin_memory=False, prefetch_factor=2,
                                persistent_workers=False)
        else:
            first = 0
            loader = DataLoader(Frames(train, args.packed, grid, class_weights), batch_size=None, shuffle=True, generator=g, num_workers=1,
                                collate_fn=_as_is, pin_memory=False, prefetch_factor=2, persistent_workers=False)  # 1 worker
        block.train()
        t0, run = time.time(), {}
        for i, (images, metas, fg) in enumerate(loader, start=first):
            if args.max_steps and i >= args.max_steps:
                break
            s, t, y, loss, parts = forward_loss(images, metas, fg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            opt.step()
            sched.step()
            run["loss"] = run.get("loss", 0.0) + float(loss)
            for k, v in parts.items():
                if math.isfinite(float(v)):  # fg split is nan on a frame without boxes
                    run[k] = run.get(k, 0.0) + float(v)
            del s, t, y, loss
            if use_layers or use_pred:  # fragmentation pushed reserved to ~7 GiB -> driver OOM at 8 GB
                torch.cuda.empty_cache()
            if args.save_every and (i + 1) % args.save_every == 0 and i + 1 < steps_per_epoch:
                torch.save({"model": block.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "epoch": epoch, "step": i + 1, "history": history, "best": best, "cfg": block.cfg},
                           out / "last_step.pt.tmp")
                (out / "last_step.pt.tmp").replace(out / "last_step.pt")
            every = 10 if args.max_steps else 100
            if (i + 1) % every == 0:
                print(f"ep {epoch} step {i + 1}/{steps_per_epoch} " + " ".join(f"{k} {v / every:.4f}" for k, v in run.items())
                      + f" lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (i + 1 - first):.2f} s/step"
                      f" peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                run = dict.fromkeys(run, 0.0)
        v = run_val()
        history.append({"epoch": epoch, "val": v, "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch} val {json.dumps(v)} ({history[-1]['minutes']:.1f} min)", flush=True)
        if v["loss"] < best:
            best = v["loss"]
            torch.save({"cfg": block.cfg, "model": block.state_dict(), "epoch": epoch, "val": v}, out / "best.pt")
        torch.save({"model": block.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "history": history, "best": best, "cfg": block.cfg}, out / "last.pt")

    # Block latency (bf16, eval mode) vs the upstream permute + conv stack, on one val frame.
    block.eval()
    fd = args.packed / val[0]
    s, _ = step_tensors(student, norm, head, load_images(fd), PerceptionFrame(fd).img_metas(image_size=IMAGE_SIZE))
    lat = {}
    with torch.no_grad():
        for name, fn in (("light_block", lambda: block(s)), ("upstream_permute_conv", lambda: teacher_encoding(head.bev_modeling.view_trans, s))):
            ms = []
            for k in range(13):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fn()
                torch.cuda.synchronize()
                if k >= 3:
                    ms.append(1000 * (time.perf_counter() - t1))
            lat[name] = float(np.median(ms))
    (out / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "params_M": n_params / 1e6,
                                                  "history": history, "best_val_loss": best, "latency_ms": lat}, indent=2))
    print(f"latency {lat}; {args.run_name} done", flush=True)


if __name__ == "__main__":
    main()
