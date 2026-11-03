"""Apply the heading refiner to exported predictions: flip pedestrian and bicycle yaw where it disagrees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qdistill.heading_crops import wrap
from train_heading_refiner import CLS_IDX, HeadingRefiner

ROOT = Path(__file__).resolve().parents[2]


def oracle_flags(r, g, match_dist=2.0):
    """{box index: should flip} for pedestrian / bicycle predictions matched to GT."""
    out = {}
    if not r["boxes"] or not g or not g["boxes"]:
        return out
    b, lab, sc = np.asarray(r["boxes"]), np.asarray(r["labels"]), np.asarray(r["scores"])
    gb, gl = np.asarray(g["boxes"]), np.asarray(g["labels"])
    for c in CLS_IDX:
        pi_, gi = np.where(lab == c)[0], np.where(gl == c)[0]
        used = np.zeros(len(gi), bool)
        for k in pi_[np.argsort(-sc[pi_])]:
            if not len(gi):
                break
            d = np.hypot(gb[gi, 0] - b[k, 0], gb[gi, 1] - b[k, 1])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= match_dist:
                used[j] = True
                out[int(k)] = bool(abs(wrap(b[k, 6] - gb[gi[j], 6])) > np.pi / 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=ROOT / "weights/heading_refiner.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--flip-labels", nargs="+", type=int, default=[4, 2],
                   help="labels that may be flipped (4 = pedestrian, 2 = bicycle); others are left alone")
    p.add_argument("--angle-mode", default="flip", choices=("flip", "replace"),
                   help="flip: yaw += pi where the refiner disagrees by > 90 deg; replace: every box "
                        "with confidence >= tau takes the refiner's angle, yaw += wrap(alpha_ref - alpha_dec) (exact for a "
                        "level camera), velocity rotated by the same angle (kept for --keep-vel labels)")
    p.add_argument("--keep-vel", nargs="*", type=int, default=[],
                   help="labels whose velocity is kept on a flip (yaw only), e.g. 2 = bicycle")
    p.add_argument("--max-speed", nargs="*", default=[], metavar="LABEL:V",
                   help="no flip for that label above predicted speed V m/s, e.g. 2:1.0 (moving bikes)")
    p.add_argument("--slow-tau", nargs="*", default=[], metavar="LABEL:V:TAU",
                   help="tau TAU for that label below predicted speed V m/s, e.g. 4:0.3:0.9 (standing peds)")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # per-class tau from the held-out set (None: the class never reaches the target accuracy -> never flipped);
    # checkpoints before tau_bike used the pedestrian tau for both
    ev = ck["eval"]
    tau_c = {c: args.tau if args.tau is not None else ev.get(key, ev["tau"])
             for c, key in ((4, "tau"), (2, "tau_bike"))}
    tau_c = {c: np.inf if t is None else t for c, t in tau_c.items()}
    model = HeadingRefiner(pretrained=False).cuda().eval()
    model.load_state_dict(ck["model"])
    xs = [np.load(f, mmap_mode="r") for f in sorted(args.crops.glob("pred_*_x.npy"))]
    ms = [np.load(str(f).replace("_x.npy", "_meta.npz")) for f in sorted(args.crops.glob("pred_*_x.npy"))]
    meta = {k: np.concatenate([m[k] for m in ms]) for k in ("alpha", "cls", "token", "box")}
    ap, cf, off = [], [], 0
    with torch.no_grad():
        for x in xs:
            for i in range(0, len(x), 512):
                xb = torch.from_numpy(np.asarray(x[i:i + 512], np.float32)).permute(0, 3, 1, 2).cuda()
                n = len(xb)
                cls = torch.tensor([CLS_IDX[int(c)] for c in meta["cls"][off:off + n]]).cuda()
                off += n
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a, c = model(xb, cls)
                ap.append(a.float().cpu().numpy()); cf.append(torch.sigmoid(c).float().cpu().numpy())
    ap, cf = np.concatenate(ap), np.concatenate(cf)
    d = json.loads(args.pred.read_text())
    speed = np.array([np.hypot(*d["results"][str(t)]["boxes"][int(k)][7:9]) for t, k in zip(meta["token"], meta["box"])])
    tau_box = np.vectorize(tau_c.get)(meta["cls"].astype(int)).astype(float)
    for r in args.slow_tau:
        c, v, t = r.split(":")
        tau_box[(meta["cls"] == int(c)) & (speed < float(v))] = float(t)
    disagree = np.abs(wrap(ap - meta["alpha"])) > np.pi / 2
    conf_ok = (cf >= tau_box) & np.isin(meta["cls"], args.flip_labels)
    for r in args.max_speed:
        c, v = r.split(":")
        conf_ok &= ~((meta["cls"] == int(c)) & (speed > float(v)))
    flip = disagree & conf_ok
    delta = np.where(flip, np.pi, 0.0) if args.angle_mode == "flip" else np.where(conf_ok, wrap(ap - meta["alpha"]), 0.0)

    g = json.loads(args.gt.read_text())["results"] if args.gt else None
    tp = fp = fn = 0
    by_tok: dict[str, list[tuple[int, float]]] = {}
    for t, k, f, dl in zip(meta["token"], meta["box"], flip, delta):
        if f or dl != 0.0:
            by_tok.setdefault(str(t), []).append((int(k), float(dl)))
    if g is not None:
        cropped = {}
        for t, k in zip(meta["token"], meta["box"]):
            cropped.setdefault(str(t), set()).add(int(k))
        for t, ks in cropped.items():
            ora = oracle_flags(d["results"][t], g.get(t))
            fl = {k for k, dl in by_tok.get(t, []) if abs(wrap(dl)) > np.pi / 2}
            for k in ks:
                if k in ora:
                    tp += int(ora[k] and k in fl); fp += int(not ora[k] and k in fl); fn += int(ora[k] and k not in fl)
    for t, ks in by_tok.items():
        r = d["results"][t]
        for k, dl in ks:
            b = r["boxes"][k]
            b[6] = float(wrap(b[6] + dl))
            if r["labels"][k] in args.keep_vel:
                continue
            if args.angle_mode == "flip":
                b[7], b[8] = -b[7], -b[8]
            else:
                c, sn = np.cos(dl), np.sin(dl)
                b[7], b[8] = float(c * b[7] - sn * b[8]), float(sn * b[7] + c * b[8])
    args.output.write_text(json.dumps(d))
    msg = (f"{args.pred.name}: {len(ap)} cropped boxes, refiner disagrees on {disagree.mean():.1%}, flipped "
           f"{flip.sum()} ({flip.mean():.1%}) at tau ped {tau_c[4]:.2f} / bike {tau_c[2]:.2f}")
    if args.angle_mode == "replace":
        msg += f"; replace: {int(conf_ok.sum())} boxes re-angled, median |delta| {np.degrees(np.median(np.abs(delta[conf_ok]))) if conf_ok.any() else 0:.1f} deg"
    if g is not None:
        msg += (f"; vs oracle (matched boxes): precision {tp / max(tp + fp, 1):.3f}, recall {tp / max(tp + fn, 1):.3f} "
                f"(tp {tp}, fp {fp}, fn {fn})")
    print(msg + f" -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
