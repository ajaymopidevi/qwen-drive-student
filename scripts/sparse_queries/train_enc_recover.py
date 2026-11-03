"""Fine-tune the BEV encoder and dense heads in deployment numerics: KD + GT map and occupancy."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qdistill.student_io import load_student
from qdistill import merged_pool, trt_modules
from qdistill.bev2d import EncoderTail, Fp16Run, Stage2Targets, heads_from_bev
from qdistill.data import load_frame_images, split_tokens
from qdistill.dense_eval import DenseScorer, load_gt
from qdistill.head_grad import enable_head_grad
from qdistill.pred_kd import pred_losses
from qdistill.teacher import load_head, load_processor
from qdistill.trt_install import TrtHead
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]
OCC_EMPTY = 9  # OCC_EMPTY_LABEL (trt_occ_refiner.EMPTY)


def nmse(s, t):
    t = t.float()
    return ((s.float() - t) ** 2).mean() / t.var().clamp_min(1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--skip-ca", default="", help="comma list of encoder layers whose spatial cross-attention is skipped")
    p.add_argument("--w-occ", type=float, default=0.0, help="occupancy KL weight (0 = occupancy not trained)")
    p.add_argument("--occ-weight", type=float, default=4.0, help="KL weight of voxels the target labels non-empty")
    p.add_argument("--occ-refiner-ckpt", type=Path, default=ROOT / "weights/occ_refiner.pt",
                   help="PyTorch occupancy refiner of the student path (qdistill.occ2d checkpoint)")
    p.add_argument("--enc-precision", choices=("bf16", "fp32"), default="bf16",
                   help="encoder forward / backward precision after the self-check (fp32 = exact math = the "
                        "TensorRT engine's numerics; PyTorch MSDA)")
    p.add_argument("--front-trt", type=Path, default=None,
                   help="engine dir; the FRONT modules run as engines in the Stage-2 targets")
    p.add_argument("--train-heads", default="", help="comma list of trt_modules.HEADS trained (fp32 "
                   "masters, fp16 autocast)")
    p.add_argument("--heads-lr", type=float, default=2e-5)
    p.add_argument("--init-heads", type=Path, default=None, help="initial --train-heads masters (final.pt[\"heads\"])")
    p.add_argument("--zhalf", type=Path, default=None,
                   help="Z 8 voxel engine (qdistill.zhalf); prior / p16 from the block at 8 height levels")
    p.add_argument("--warp-grid", type=Path, default=None,
                   help="warped-grid voxel engine (G x G, qdistill.warp_grid); student path on the warped grid")
    p.add_argument("--warp-b", type=float, default=25.6, help="fine-zone half width (m), 51.2 - 1.024 k")
    p.add_argument("--warp-z8", type=Path, default=None,
                   help="zhalf v2 engine (G x G, 16 levels in, 8 out) in place of --warp-grid's block")
    p.add_argument("--init-enc", type=Path, default=None, help="E: initial layers (train_enc_recover final.pt)")
    p.add_argument("--w-gt-map", type=float, default=0.0)
    p.add_argument("--w-gt-occ", type=float, default=0.0)
    p.add_argument("--dense-eval-at", default="", help="comma list of steps (0 = before training)")
    p.add_argument("--dense-val-split", default="val_quick")
    p.add_argument("--dense-val-stride", type=int, default=3)
    p.add_argument("--dense-val-frames", type=int, default=0, help="cap (smoke)")
    p.add_argument("--poses", nargs=2, type=Path,
                   default=[ROOT / "data/gt/ego_poses_train.json", ROOT / "data/gt/ego_poses_val.json"])
    p.add_argument("--train-from", type=int, default=0, help="layers below this index are frozen (memory fallback)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--w-bev", type=float, default=1.0)
    p.add_argument("--w-det", type=float, default=1.0)
    p.add_argument("--w-map", type=float, default=1.0)
    p.add_argument("--ckpt-heads", action="store_true", help="activation checkpointing in the frozen heads")
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
    sw = blk = None
    if args.zhalf is not None:  # the block at 8 height levels; grid, queries, targets unchanged
        from qdistill import zhalf
        assert blk is None, "--zhalf is used alone"
        blk = zhalf.Switch(head, args.zhalf)
    if args.warp_grid is not None:
        from qdistill import warp_grid
        assert sw is None and blk is None, "--warp-grid is used alone"
        assert not (args.front_trt and {"uvtr_query_proj", "uvtr_occ_proj"} & set(names)), \
            "warp grid: the UVTR projections must stay eager (their engines are 200 x 200)"
        sw = warp_grid.Switch(head, args.warp_grid, args.warp_b, z8_engine=args.warp_z8)
        print(f"warp grid: b {args.warp_b} m -> {sw.w.G} x {sw.w.G}, decoder hooks {sw.n_hooks}", flush=True)
        if args.warp_z8 is not None:
            print(f"warp z8: block engine {args.warp_z8} (8 levels)", flush=True)
    half = sw.half if sw is not None else nullcontext

    def prior_tokens(v):  # capture the clean prior [Y * X, 256] (Stage2Targets resets cap before each frame)
        t = f_prior(v)
        targets.cap["prior"] = t.detach()
        return t
    bev._uvtr_voxel_to_bev_tokens = prior_tokens
    enable_head_grad()
    processor = load_processor(args.weights)
    scene_of = {}
    for f in args.poses:
        scene_of.update({t: v["scene"] for t, v in json.loads(f.read_text()).items()})
    hd = bev.head
    n_layers = len(hd.transformer.encoder.layers)
    L = args.train_from
    front = EncoderTail(head, L, first=0).eval().requires_grad_(False) if L else None
    back = EncoderTail(head, n_layers - L, first=L)
    refiner = None
    if args.w_occ > 0:
        from qdistill import occ2d
        ck = torch.load(args.occ_refiner_ckpt, map_location="cpu", weights_only=False)
        refiner = occ2d.build(ck["cfg"])
        refiner.load_state_dict(ck["model"])
        refiner = refiner.cuda().to(torch.bfloat16).eval().requires_grad_(False)
    hnames = [n for n in args.train_heads.split(",") if n.strip()]
    PROJ = ("uvtr_query_proj", "uvtr_occ_proj")  # the UVTR projections, trained on the volume kept by the switch
    assert all(n in trt_modules.HEADS + PROJ for n in hnames), hnames
    if set(PROJ) & set(hnames):
        assert sw is not None and getattr(sw, "z8_engine", None), "trainable UVTR projections need --warp-z8"
        sw.keep_volume = True
    assert refiner is not None or not {"uvtr_occ_fuse", "occ_pred_head"} & set(hnames), "occupancy heads need --w-occ"
    hmods = {n: Fp16Run(trt_modules.fp32_copy(trt_modules.get(bev, trt_modules.TARGETS[n])).cuda().requires_grad_(True))
             for n in hnames}  # eval mode (BN statistics frozen), weights trained
    if args.init_heads is not None:
        hck = torch.load(args.init_heads, map_location="cpu", weights_only=False)["heads"]
        for n, m in hmods.items():
            if n not in hck:  # a projection the init run did not train keeps the head's own weights
                print(f"head {n} not in {args.init_heads}: kept the head's weights", flush=True)
                continue
            m.m.load_state_dict({k: v.float() for k, v in hck[n].items()})
        print(f"heads {sorted(hmods)} initialised from {args.init_heads}", flush=True)
    groups = [{"params": list(back.parameters()) + ([sw.queries] if sw is not None else [])}]
    if hmods:
        groups.append({"params": [q for m in hmods.values() for q in m.parameters()], "lr": args.heads_lr})
    params = [q for g in groups for q in g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    train = split_tokens(args.splits, args.train_split)[:: args.train_stride]
    val = split_tokens(args.splits, args.val_split)[:: args.val_stride][: args.val_frames]
    print(f"train {args.run_name}: skip_ca [{args.skip_ca}], "
          f"w det {args.w_det} map {args.w_map} occ {args.w_occ} (non-empty x{args.occ_weight}), "
          f"layers {L}..{n_layers - 1} trained, heads {hnames or 'frozen'} "
          f"({sum(q.numel() for q in back.parameters()) / 1e6:.2f}M params), {args.steps} steps of {len(train)} frames, "
          f"val {len(val)} frames", flush=True)

    def stage2(t, to_half=True):
        metas = processor(PerceptionFrame(args.packed / t))[1]
        images = load_frame_images(args.packed, t, "jpeg_gpu")[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vit_n, llm_n = student(images)
        c = targets(norm.denormalize("vit", vit_n[0]), norm.denormalize("llm", llm_n[0]), metas)
        if sw is not None and to_half:
            sw.to_half(c, metas, f_prior)
        if blk is not None and to_half:
            blk.to_block(c, metas, f_prior)
        return c, metas

    def encode(prior, c, metas):
        emb = sw.queries if sw is not None and hd.bev_h != sw.full_st["bev_h"] else hd.bev_embedding.weight
        q = emb.float() + prior.float()                                            # [Y * X, 256]
        q = q.T.reshape(1, -1, hd.bev_h, hd.bev_w)
        if front is not None:
            with torch.no_grad():
                q = front(q, c["llm"], metas)
        return back(q, c["llm"], metas)

    def proj_inputs(c):
        """prior / p16 recomputed with grad from the switch's 16-level volume by the trained projections."""
        y, c = c["y"], dict(c)
        if "uvtr_query_proj" in hmods:
            p = hmods["uvtr_query_proj"](y.reshape(y.shape[0], -1, *y.shape[-2:]))  # as bev._uvtr_voxel_to_bev_tokens
            c["prior"] = p.squeeze(0).permute(1, 2, 0).reshape(-1, p.shape[1])
        if "uvtr_occ_proj" in hmods:
            c["p16"] = hmods["uvtr_occ_proj"](y).flatten(1, 2)
        return c

    def student_path(c, metas, occ, ckpt=False):
        """encoder -> heads: (bev, head outputs)."""
        if "y" in c:
            c = proj_inputs(c)
        with half():
            b = encode(c["prior"].float(), c, metas)
            return b, heads_from_bev(bev, b, c["p16"], metas, refiner, occ=occ, ckpt=ckpt, mods=hmods)
        return b, heads_from_bev(bev, b, c["p16"], metas, refiner, occ=occ, ckpt=ckpt, mods=hmods)

    def losses(t):
        c, metas = stage2(t)
        b, s = student_path(c, metas, refiner is not None, args.ckpt_heads)
        Lo = pred_losses(s, c["outs"])
        if refiner is not None:  # non-empty voxels (target argmax) weighted --occ-weight, as distill_refiner.Losses
            ls, lt = s["occ_pred"].float(), c["outs"]["occ_pred"].float()        # [1, X, Y, Z, K]
            kl = (lt.softmax(-1) * (lt.log_softmax(-1) - ls.log_softmax(-1))).sum(-1)
            w = 1.0 + (args.occ_weight - 1.0) * (lt.argmax(-1) != OCC_EMPTY).float()
            Lo["occ"] = (kl * w).sum() / w.sum()
        Lo["feat_bev"] = nmse(b, c["bev"])
        loss = args.w_bev * Lo["feat_bev"] + args.w_det * (Lo["det_cls"] + Lo["det_box"]) + args.w_map * Lo["map"]
        if refiner is not None:
            loss = loss + args.w_occ * Lo["occ"]
        if (args.w_gt_map or args.w_gt_occ) and (ROOT / "data/gt/map" / f"{t}.npy").exists():  # arm E
            gt = load_gt(t, scene_of[t])                                     # (val_rest KD frames have no map GT)
            Lo["map_gt"] = F.cross_entropy(s["seg_preds"].float(), torch.from_numpy(gt["map"]).cuda().long()[None])
            cam = torch.from_numpy(gt["cam"]).cuda()
            Lo["occ_gt"] = F.cross_entropy(s["occ_pred"][0].float()[cam], torch.from_numpy(gt["occ10"]).cuda().long()[cam])
            loss = loss + args.w_gt_map * Lo["map_gt"] + args.w_gt_occ * Lo["occ_gt"]
        return loss, Lo, s

    @torch.no_grad()
    def validate():
        back.eval()
        acc = {}
        for t in val:
            loss, Lo, _ = losses(t)
            for k, v in {"loss": loss, **Lo}.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        back.train()
        return {k: round(v / len(val), 5) for k, v in acc.items()}

    # self-check: the copied encoder on the clean prior reproduces the Stage-2 bev_embed (teacher encoder only)
    with torch.no_grad():
        back.eval()
        chk = []
        for t in val[: args.check_frames]:
            c, metas = stage2(t, to_half=False)
            chk.append(float(nmse(encode(c["prior"], c, metas), c["bev"])))
    print(f"self-check: clean-prior encoder vs Stage-2 bev_embed nMSE {chk}", flush=True)
    assert max(chk) < 1e-3, f"encoder copy does not reproduce bev_embed: {chk}"
    skip = sorted(int(s) for s in args.skip_ca.split(",") if s.strip())
    for i in skip:  # attentions[0] = temporal self-attention, [1] = spatial cross-attention (head_chain)
        tail, j = (front, i) if i < L else (back, i - L)
        tail.enc.layers[j].attentions[1].forward = lambda query, *a, **k: query
    if skip:
        print(f"spatial cross-attention skipped in layers {skip}", flush=True)
    if args.init_enc is not None:  # E: start from a fine-tuned checkpoint (after the self-check)
        ck = torch.load(args.init_enc, map_location="cpu", weights_only=False)
        for i, sd in ck["layers"].items():
            back.enc.layers[int(i) - L].load_state_dict({k: v.float() for k, v in sd.items()})
        print(f"encoder layers {sorted(int(i) for i in ck['layers'])} initialised from {args.init_enc}", flush=True)
        if args.warp_grid is not None and "warp_grid" in ck:  # and the trained warped BEV queries
            assert abs(ck["warp_grid"]["b"] - args.warp_b) < 1e-6, (ck["warp_grid"]["b"], args.warp_b)
            sw.queries.data.copy_(ck["warp_grid"]["bev_embedding"].float())
            print(f"warped BEV queries initialised from {args.init_enc}", flush=True)
    if sw is not None:
        sw.retarget(back)
        print("warped grid: BEV queries trained", flush=True)
    if args.enc_precision == "fp32":  # after the self-check (bf16 reproduces the Stage-2 bev_embed, fp32 does not)
        assert front is None, "--enc-precision fp32: all layers trained (no frozen front)"
        back.dtype = torch.float32
        print("encoder runs in fp32 (exact math, PyTorch MSDA)", flush=True)
    dense_at = {int(x) for x in args.dense_eval_at.split(",") if x.strip()}
    dval = split_tokens(args.splits, args.dense_val_split)[:: args.dense_val_stride][: args.dense_val_frames or None]

    @torch.no_grad()
    def dense_eval(step):
        back.eval()
        sc, t1 = DenseScorer(), time.time()
        for t in dval:
            c, metas = stage2(t)
            _, s = student_path(c, metas, True)
            sc.add(scene_of[t], s["seg_preds"][0].argmax(0).byte().cpu().numpy(),
                   s["occ_pred"][0].argmax(-1).byte().cpu().numpy(), load_gt(t, scene_of[t]))
        back.train()
        sc.save(run / f"dense_{step}.json", {"step": step, "run": args.run_name})
        print(f"dense eval step {step} ({len(dval)} frames, {time.time() - t1:.0f} s) {json.dumps(sc.summary())}", flush=True)

    history = {"args": {k: str(v) for k, v in vars(args).items()}, "self_check_nmse": chk, "skip_ca": skip}
    v0 = validate()
    history["val_zero_shot"] = v0
    print(f"val zero-shot (blurred prior, teacher encoder) {json.dumps(v0)}", flush=True)
    if 0 in dense_at:
        dense_eval(0)
    back.train()
    order = np.random.default_rng(args.seed).permutation(len(train))
    t0, run_l = time.time(), {}
    for k in range(args.steps):
        loss, Lo, _ = losses(train[order[k % len(train)]])
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
        if k + 1 in dense_at:
            dense_eval(k + 1)
    v1 = validate()
    history.update(val_fine_tuned=v1, train={a: round(b / args.steps, 5) for a, b in run_l.items()},
                   min=round((time.time() - t0) / 60, 1), peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(f"val fine-tuned {json.dumps(v1)}", flush=True)
    torch.save({"layers": {L + i: {k: v.detach().cpu() for k, v in lay.state_dict().items()}
                           for i, lay in enumerate(back.enc.layers)},
                **({"heads": {n: {k: v.detach().cpu() for k, v in m.m.state_dict().items()} for n, m in hmods.items()}}
                   if hmods else {}),
                **({"warp_grid": sw.state()} if args.warp_grid is not None else {}),
                **({"zhalf": {"voxel_engine": str(args.zhalf)}} if args.zhalf is not None else {}),
                "args": history["args"]}, run / "final.pt")
    (run / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done {args.run_name} -> {run / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
