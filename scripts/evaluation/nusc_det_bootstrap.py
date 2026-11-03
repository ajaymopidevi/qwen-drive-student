"""Paired scene bootstrap of devkit mAP and NDS for several prediction files (eval env)."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nusc_det_eval import (CLASS_RANGE, DIST_TH_TP, DIST_THS, MIN_PRECISION, MIN_RECALL, TP_EXCLUDED, TP_USED,
                           SlimNuScenes, to_eval_boxes)
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.common.utils import center_distance, cummean, scale_iou, velocity_l2, yaw_diff
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionMetricData

ROOT = Path(__file__).resolve().parents[2]
ERR_FNS = {"trans_err": center_distance, "vel_err": velocity_l2,
           "scale_err": lambda g, p: 1 - scale_iou(g, p)}


def match_records(gt, pred, classes):
    """records[(cls, th)][sample] = (conf, tp, global_idx) arrays (+ errs at DIST_TH_TP); npos[cls][sample]."""
    rec = defaultdict(dict)
    npos = defaultdict(dict)
    gidx = {id(b): i for i, b in enumerate(pred.all)}  # devkit tie-break: (conf, index) sorted, then reversed
    for s in gt.sample_tokens:
        g_all, p_all = gt[s], pred[s]
        for c in classes:
            g = [b for b in g_all if b.detection_name == c]
            p = sorted((b for b in p_all if b.detection_name == c), key=lambda b: (b.detection_score, gidx[id(b)]),
                       reverse=True)
            npos[c][s] = len(g)
            gxy = np.array([b.translation[:2] for b in g]).reshape(-1, 2)
            pxy = np.array([b.translation[:2] for b in p]).reshape(-1, 2)
            dist = np.linalg.norm(pxy[:, None] - gxy[None], axis=2) if len(g) and len(p) else np.zeros((len(p), len(g)))
            for th in DIST_THS:
                taken = np.zeros(len(g), dtype=bool)
                tp = np.zeros(len(p), dtype=bool)
                errs = {k: [] for k in ("trans_err", "vel_err", "scale_err", "orient_err")}
                for i, pb in enumerate(p):
                    d = np.where(taken, np.inf, dist[i]) if len(g) else np.array([np.inf])
                    j = int(np.argmin(d))
                    if d[j] < th:
                        taken[j] = True
                        tp[i] = True
                        if th == DIST_TH_TP:
                            gb = g[j]
                            for k, fn in ERR_FNS.items():
                                errs[k].append(fn(gb, pb))
                            errs["orient_err"].append(yaw_diff(gb, pb, period=np.pi if c == "barrier" else 2 * np.pi))
                rec[(c, th)][s] = (np.array([b.detection_score for b in p]), tp,
                                   np.array([gidx[id(b)] for b in p], dtype=np.int64),
                                   {k: np.array(v, dtype=float) for k, v in errs.items()} if th == DIST_TH_TP else None)
    return rec, npos


def metric_data(conf, tp, idx, errs, n_pos) -> DetectionMetricData:
    """Devkit accumulate() from per-prediction match results (same sorting, interpolation and cummean)."""
    if n_pos == 0 or tp.sum() == 0:
        return DetectionMetricData.no_predictions()
    order = np.lexsort((idx, conf))[::-1]  # sorted((conf, index)) reversed
    conf, tp = conf[order], tp[order]
    tpc, fpc = np.cumsum(tp).astype(float), np.cumsum(~tp).astype(float)
    prec, rec = tpc / (fpc + tpc), tpc / float(n_pos)
    r = np.linspace(0, 1, DetectionMetricData.nelem)
    prec_i = np.interp(r, rec, prec, right=0)
    conf_i = np.interp(r, rec, conf, right=0)
    out = {}
    if errs is not None:
        mconf = conf[tp]  # matched confidences in global order; errs (matched_errors) are in the same order
        for k, v in errs.items():
            tmp = cummean(v)
            out[k] = np.interp(conf_i[::-1], mconf[::-1], tmp[::-1])[::-1]
    else:
        out = {k: np.ones(DetectionMetricData.nelem) for k in ("trans_err", "vel_err", "scale_err", "orient_err")}
    return DetectionMetricData(recall=r, precision=prec_i, confidence=conf_i, attr_err=np.ones(DetectionMetricData.nelem), **out)


def evaluate(rec, npos, scenes_of, draw, classes):
    """mAP, NDS, per-class AP for one draw (list of scene ids with repetition)."""
    samples = [s for sc in draw for s in scenes_of[sc]]
    seen = defaultdict(int)
    occ = []  # 0 for a sample's first copy in the draw, 1 for the second, ... (devkit tie-break index stays intact)
    for s in samples:
        occ.append(seen[s])
        seen[s] += 1
    per_class = {}
    for c in classes:
        n = sum(npos[c][s] for s in samples)
        if n == 0:
            continue
        aps, tps = [], {}
        for th in DIST_THS:
            parts = [rec[(c, th)][s] for s in samples]
            conf = np.concatenate([q[0] for q in parts])
            tp = np.concatenate([q[1] for q in parts])
            # Tie-break index = devkit's original index, offset only for repeated copies of a sample.
            idx = np.concatenate([q[2] + o * 10_000_000 for o, q in zip(occ, parts)])
            errs = None
            if th == DIST_TH_TP:
                errs = matched_errors(parts, occ)  # re-ordered like the global (conf, index) sort
            md = metric_data(conf, tp, idx, errs, n)
            aps.append(calc_ap(md, MIN_RECALL, MIN_PRECISION))
            if th == DIST_TH_TP:
                tps = {m: float("nan") if m in TP_EXCLUDED.get(c, ()) else float(calc_tp(md, MIN_RECALL, m)) for m in TP_USED}
        per_class[c] = {"AP": float(np.mean(aps)), **tps}
    m_ap = float(np.mean([r["AP"] for r in per_class.values()]))
    tp_mean = {m: float(np.nanmean([r[m] for r in per_class.values()])) for m in TP_USED}
    nds = (5 * m_ap + sum(1 - min(1.0, e) for e in tp_mean.values())) / 10
    return {"mAP": m_ap, "NDS": nds, "tp_errors": tp_mean, "AP": {c: r["AP"] for c, r in per_class.items()}}


def matched_errors(parts, occ):
    """Per-prediction errors of the matched predictions, in the global (conf, index)-descending order."""
    m_conf, m_idx, vals = [], [], defaultdict(list)
    for o, q in zip(occ, parts):
        c, t, ix, e = q
        m_conf.append(c[t])
        m_idx.append(ix[t] + o * 10_000_000)
        for name, v in e.items():
            vals[name].append(v)
    m_conf, m_idx = np.concatenate(m_conf), np.concatenate(m_idx)
    order = np.lexsort((m_idx, m_conf))[::-1]
    return {name: np.concatenate(v)[order] for name, v in vals.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataroot", type=Path, default=ROOT / "data/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gt_data = json.loads(args.gt.read_text())
    tokens = sorted(gt_data["results"])
    nusc = SlimNuScenes(args.dataroot, args.version, tokens)
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, to_eval_boxes(nusc, gt_data, tokens, 10**6)), CLASS_RANGE)
    scenes_of = defaultdict(list)
    for t in tokens:
        scenes_of[nusc.get("sample", t)["scene_token"]].append(t)
    scene_ids = sorted(scenes_of)
    classes = [c for c in CLASS_RANGE if any(b.detection_name == c for b in gt.all)]

    sets = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        pred = filter_eval_boxes(nusc, add_center_dist(nusc, to_eval_boxes(nusc, data, tokens, 500)), CLASS_RANGE)
        sets[name] = match_records(gt, pred, classes)
        del data, pred  # 800 MB full-val JSONs: free one set's boxes before parsing the next (OOM at 6 GB)
        gc.collect()
        print(f"{name}: matched", flush=True)

    full = {name: evaluate(r, n, scenes_of, scene_ids, classes) for name, (r, n) in sets.items()}
    for name, f in full.items():
        print(f"full set {name}: mAP {f['mAP']:.4f} NDS {f['NDS']:.4f}")

    rng = np.random.default_rng(args.seed)
    samples = defaultdict(list)
    for _ in range(args.reps):
        draw = list(rng.choice(scene_ids, size=len(scene_ids), replace=True))
        for name, (r, n) in sets.items():
            res = evaluate(r, n, scenes_of, draw, classes)
            samples[name].append(res)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = {"gt": str(args.gt), "reps": args.reps, "scenes": len(scene_ids), "seed": args.seed, "full": full, "ci95": {}, "paired": {}}
    names = list(sets)
    for name in names:
        out["ci95"][name] = {k: ci([s[k] for s in samples[name]]) for k in ("mAP", "NDS")}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = {}
            for k in ("mAP", "NDS"):
                d = np.array([sa[k] - sb[k] for sa, sb in zip(samples[a], samples[b])])
                pair[k] = {"diff_full": full[a][k] - full[b][k], "ci95": ci(d), "p_a_better": float((d > 0).mean())}
            pair["AP_diff"] = {c: {"diff_full": full[a]["AP"][c] - full[b]["AP"][c],
                                   "ci95": ci([sa["AP"].get(c, np.nan) - sb["AP"].get(c, np.nan)
                                               for sa, sb in zip(samples[a], samples[b])])} for c in full[a]["AP"]}
            out["paired"][f"{a}-{b}"] = pair
    args.output.write_text(json.dumps(out, indent=2))
    for name in names:
        print(f"{name}: mAP 95% CI {out['ci95'][name]['mAP']}, NDS 95% CI {out['ci95'][name]['NDS']}")
    for key, pair in out["paired"].items():
        for k in ("mAP", "NDS"):
            v = pair[k]
            print(f"{key} {k}: diff {v['diff_full']:+.4f}, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
                  f"P(first better) {v['p_a_better']:.3f}")
        for c, v in pair["AP_diff"].items():
            print(f"   {c:15s} AP diff {v['diff_full']:+.4f}  CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
