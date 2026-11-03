"""Refit the query camera sampler on top of the dense path: GT + KD."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from qdistill.student_io import load_student
from qdistill import merged_pool, trt_modules
from qdistill.bev2d import EncoderTail, Stage2Targets, heads_from_bev
from qdistill.data import load_frame_images, split_tokens
from qdistill.det_gt_loss import gt_losses, gt_targets
from qdistill.head_grad import enable_head_grad
from qdistill.pred_kd import pred_losses
from qdistill.query_sampling import SamplingDecoder, adapt_state, vit_levels
from qdistill.teacher import load_head, load_processor
from qdistill.trt_install import TrtHead
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--dense", required=True, help="noca:<train_enc_recover ckpt> or conv:<train_enc_recover --encoder conv ckpt>")
    p.add_argument("--w-gt", type=float, default=0.0, help="GT detection loss weight (0 = KD only)")
    p.add_argument("--n-learn", type=int, default=6, help="learned keypoints per query (+ 7 fixed box points)")
    p.add_argument("--sample-layers", default="0,1,2,3,4,5")
    p.add_argument("--levels", choices=("llm", "vit", "vit3"), default="llm",
                   help="sampler input: LLM-adaptor levels or the ViT-neck output (+ its 2x2 / 4x4 pools)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4, help="samplers")
    p.add_argument("--lr-dec", type=float, default=2e-5, help="teacher decoder layers (0 = frozen)")
    p.add_argument("--enc-precision", choices=("bf16", "fp32"), default="bf16",
                   help="run the frozen noca dense path in fp32")
    p.add_argument("--front-trt", type=Path, default=None,
                   help="engine dir; the frozen FRONT modules run as engines (the deployed prior)")
    p.add_argument("--init-sampler", type=Path, default=None,
                   help="start from a trained SamplingDecoder (final.pt), loaded after the self-check")
    p.add_argument("--warp-offsets", action="store_true",
                   help="warp-grid checkpoint; decoder sampling offsets in metric (warp_grid.enable_offsets)")
    p.add_argument("--half-grid", type=Path, default=None, help="100 x 100 voxel engine (half-grid --dense)")
    p.add_argument("--half-block", type=Path, default=None, help="100 x 100 voxel engine (default: the checkpoint's)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--ckpt-heads", action="store_true", help="activation checkpointing of the decoder")
    p.add_argument("--train-split", default="s1_C4k")
    p.add_argument("--val-split", default="val_rest")
    p.add_argument("--val-stride", type=int, default=40)
    p.add_argument("--val-frames", type=int, default=30)
    p.add_argument("--check-frames", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/student.pt", help="Stage-1 student")
    p.add_argument("--voxel-engine", type=Path, default=ROOT / "weights/engines/voxel_r64_fp16.engine")
    p.add_argument("--refiner-engine", type=Path, default=ROOT / "weights/engines/occ_refiner_fp16.engine")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-trainval.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-trainval")
    p.add_argument("--cache", type=Path, default=ROOT / "cache/teacher/v1.0-trainval")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Qwen-Drive-1.0-4B")
    p.add_argument("--warp-block", type=Path, default=None, help="block engine, in place of the path in the --dense checkpoint")
    p.add_argument("--warp-z8", type=Path, default=None, help="Z 8 block engine, in place of the path in the --dense checkpoint")
    args = p.parse_args()

    run = ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    student, norm = load_student(args.ckpt, args.cache)
    student.requires_grad_(False)
    head = load_head(args.weights)
    head.requires_grad_(False)
    bev = head.bev_modeling
    trt = TrtHead(head, args.voxel_engine, args.refiner_engine)
    merged_pool.install(head, trt)
    if args.front_trt is not None:
        names = [n for n in trt_modules.FRONT if trt_modules.engine_path(args.front_trt, n).exists()]
        print("front modules as engines:", trt_modules.install(head, args.front_trt, names), flush=True)
    targets = Stage2Targets(head)
    f_prior = bev._uvtr_voxel_to_bev_tokens

    def prior_tokens(v):  # capture the clean prior [Y * X, 256] (Stage2Targets resets cap before each frame)
        t = f_prior(v)
        targets.cap["prior"] = t.detach()
        return t
    bev._uvtr_voxel_to_bev_tokens = prior_tokens
    enable_head_grad()
    processor = load_processor(args.weights)
    hd = bev.head
    tr = hd.transformer

    kind, _, path = args.dense.partition(":")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "noca":
        enc = EncoderTail(head, len(tr.encoder.layers), first=0).eval().requires_grad_(False)
        if "enc_layers" in ck:  # truncated encoder
            enc.enc.layers = enc.enc.layers[: ck["enc_layers"]]
        for i, sd in ck["layers"].items():
            enc.enc.layers[int(i)].load_state_dict(sd)
        for lay in enc.enc.layers:  # attentions[1] = spatial cross-attention (head_chain skip_ca)
            lay.attentions[1].forward = lambda query, *a, **k: query
        if args.enc_precision == "fp32":
            enc.dtype = torch.float32
    else:
        raise ValueError(f"--dense {args.dense!r}")
    sw = None
    if "warp_grid" in ck:  # dense path + decoder on the warped grid (its own G x G voxel engine)
        from qdistill import warp_grid
        assert kind == "noca" and sw is None, "warp grid checkpoint"
        wg = ck["warp_grid"]
        sw = warp_grid.Switch(head, args.warp_block or wg["voxel_engine"], wg["b"], z8_engine=args.warp_z8 or wg.get("z8_engine"))  # v2 block
        sw.queries.data.copy_(wg["bev_embedding"].float())
        sw.retarget(enc)
        print(f"warp grid: b {wg['b']} m, {sw.w.G} x {sw.w.G}", flush=True)
        if args.warp_offsets:
            print(f"warp offsets: {warp_grid.enable_offsets(head)} decoder attentions", flush=True)
    assert not args.warp_offsets or "warp_grid" in ck, "--warp-offsets needs a warp-grid checkpoint"
    f_dense_prior = f_prior
    if "uvtr_query_proj" in ck.get("heads", {}):  # the dense path's prior from the fine-tune's trained projection
        from qdistill.bev2d import Fp16Run          # (the head's own module keeps the teacher's: Stage-2 targets unchanged)
        assert sw is not None, "trained uvtr_query_proj: warp-grid checkpoint only"
        qp = Fp16Run(trt_modules.fp32_copy(bev.uvtr_query_proj, ck["heads"]["uvtr_query_proj"])).cuda().eval()

        def f_dense_prior(v):                       # as bev._uvtr_voxel_to_bev_tokens
            p = qp(v.reshape(v.shape[0], -1, *v.shape[-2:]))
            return p.squeeze(0).permute(1, 2, 0).reshape(-1, p.shape[1])
        print("dense prior from the trained uvtr_query_proj", flush=True)
    blk = None
    if "zhalf" in ck:  # the voxel block at 8 height levels (its output back at 16 -> prior / p16)
        from qdistill import zhalf
        assert sw is None, "zhalf on the warped grid"
        blk = zhalf.Switch(head, ck["zhalf"]["voxel_engine"])
        print("zhalf: voxel block at 8 height levels", flush=True)
    half = sw.half if sw is not None else nullcontext

    @torch.no_grad()
    def dense(c, metas):
        with half():
            emb = sw.queries if sw is not None else hd.bev_embedding.weight
            q = (emb.float() + c["prior"].float()).T.reshape(1, -1, hd.bev_h, hd.bev_w)
            b = enc(q, c["llm"], metas)
        return b

    dec_grid = half

    dec0 = tr.decoder
    layers = [int(s) for s in args.sample_layers.split(",") if s.strip()]
    dec = SamplingDecoder(head, n_learn=args.n_learn, sample_layers=layers, levels=args.levels)

    def levels_of(c):  # what the samplers read this frame
        return c["llm"] if args.levels == "llm" else vit_levels(c["vit"], args.levels)
    lay_p, smp_p = dec.trainable(decoder_layers=args.lr_dec > 0)
    if args.lr_dec <= 0:
        dec.core.layers.requires_grad_(False)
    groups = [{"params": smp_p, "lr": args.lr}] + ([{"params": lay_p, "lr": args.lr_dec}] if lay_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    train = split_tokens(args.splits, args.train_split)[:: args.train_stride]
    val = split_tokens(args.splits, args.val_split)[:: args.val_stride][: args.val_frames]
    print(f"train {args.run_name}: dense {args.dense}, w_gt {args.w_gt}, sampler layers {layers}, keypoints 7 + {args.n_learn}, "
          f"lr {args.lr} (samplers, {sum(q.numel() for q in smp_p) / 1e6:.2f}M) / {args.lr_dec} (decoder layers, "
          f"{sum(q.numel() for q in lay_p) / 1e6:.2f}M), {args.steps} steps of {len(train)} frames, val {len(val)} frames",
          flush=True)

    def stage2(t):
        metas = processor(PerceptionFrame(args.packed / t))[1]
        images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        tr.decoder = dec0  # targets run the same head: teacher decoder, not the one being trained
        c = targets(norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0]), metas)
        if sw is not None:
            sw.to_half(c, metas, f_dense_prior)
        if blk is not None:
            blk.to_block(c, metas, f_prior)
        return c, metas

    def losses(t):
        c, metas = stage2(t)
        b = dense(c, metas)
        with dec_grid():
            dec.levels = levels_of(c)
            tr.decoder = dec
            s = heads_from_bev(bev, b, c["p16"], metas, None, occ=False, ckpt=args.ckpt_heads)
        Lo = pred_losses(s, c["outs"])
        Lo.pop("map", None)  # frozen dense path: map is 's
        loss = Lo["det_cls"] + Lo["det_box"]
        if args.w_gt > 0:
            Lo.update(gt_losses(s["all_cls_scores"], s["all_bbox_preds"], gt_targets(args.packed / t)))
            loss = loss + args.w_gt * (Lo["gt_cls"] + Lo["gt_box"])
        return loss, Lo

    @torch.no_grad()
    def validate():
        dec.eval()
        acc = {}
        for t in val:
            loss, Lo = losses(t)
            for k, v in {"loss": loss, **Lo}.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        dec.train()
        return {k: round(v / len(val), 5) for k, v in acc.items()}

    # self-check: the untrained SamplingDecoder (zero output projections) reproduces the teacher decoder
    chk = []
    with torch.no_grad():
        dec.eval()
        for t in val[: args.check_frames]:
            c, metas = stage2(t)
            b = dense(c, metas)
            with dec_grid():
                tr.decoder = dec0
                r = heads_from_bev(bev, b, c["p16"], metas, None, occ=False)
                dec.levels = levels_of(c)
                tr.decoder = dec
                s = heads_from_bev(bev, b, c["p16"], metas, None, occ=False)
            chk.append(max(float((s[k].float() - r[k].float()).abs().max()) for k in ("all_cls_scores", "all_bbox_preds")))
    print(f"self-check: untrained SamplingDecoder vs teacher decoder, max abs diff (logits, boxes) {chk}", flush=True)
    assert max(chk) < 1e-2, f"SamplingDecoder does not reproduce the teacher decoder: {chk}"

    if args.init_sampler is not None:
        ck_s = torch.load(args.init_sampler, map_location="cpu", weights_only=False)
        if ck_s["cfg"] == dec.cfg:
            dec.core.load_state_dict(ck_s["core"])  # in place: the optimizer keeps the same tensors
            print(f"SamplingDecoder initialised from {args.init_sampler}", flush=True)
        else:  # an "llm" sampler -> a "vit*" one (level rows at the ViT strides kept)
            dec.core.load_state_dict(adapt_state(ck_s["core"], ck_s["cfg"], dec))
            print(f"SamplingDecoder initialised from {args.init_sampler} (adapted llm -> {args.levels})", flush=True)
    history = {"args": {k: str(v) for k, v in vars(args).items()}, "self_check_max_abs": chk}
    v0 = validate()
    history["val_zero_shot"] = v0
    print(f"val untrained (dense path + teacher decoder) {json.dumps(v0)}", flush=True)
    dec.train()
    params = smp_p + lay_p
    order = np.random.default_rng(args.seed).permutation(len(train))
    t0, run_l = time.time(), {}
    for k in range(args.steps):
        loss, Lo = losses(train[order[k % len(train)]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        sched.step()
        for kk, vv in {"loss": loss.item(), **{a: b.item() for a, b in Lo.items()}}.items():
            run_l[kk] = run_l.get(kk, 0.0) + vv
        if (k + 1) % args.log_every == 0 or k + 1 == args.steps:
            print(f"  step {k + 1}/{args.steps} " + " ".join(f"{a} {b / (k + 1):.4f}" for a, b in run_l.items())
                  + f" lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (k + 1):.2f} s/step"
                  + f" peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    v1 = validate()
    history.update(val_fine_tuned=v1, train={a: round(b / args.steps, 5) for a, b in run_l.items()},
                   min=round((time.time() - t0) / 60, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(f"val fine-tuned {json.dumps(v1)}", flush=True)
    torch.save({**dec.state(), "args": history["args"]}, run / "final.pt")
    (run / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done {args.run_name} -> {run / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
