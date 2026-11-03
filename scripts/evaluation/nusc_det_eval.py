"""Official nuScenes detection metrics on the 7-class taxonomy (eval env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.detection import constants
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qdistill.json_stream import iter_json_array  # noqa: E402  (stdlib only; no teacher deps in this env)
# Devkit ranges; vehicle / bicycle take their source classes' range (car..construction 50 m, bicycle/motorcycle 40 m).
CLASS_RANGE = {"vehicle": 50, "czone_sign": 30, "bicycle": 40, "generic_object": 30, "pedestrian": 40,
               "traffic_cone": 30, "barrier": 30}
TP_USED = ("trans_err", "scale_err", "orient_err", "vel_err")
TP_EXCLUDED = {"traffic_cone": ("orient_err", "vel_err"), "barrier": ("vel_err",)}  # devkit class-specific definitions
DIST_THS = (0.5, 1.0, 2.0, 4.0)
DIST_TH_TP = 2.0
MIN_RECALL = MIN_PRECISION = 0.1

# DetectionBox / DetectionConfig validate names against this list object; swap its contents in place.
constants.DETECTION_NAMES[:] = list(CLASS_RANGE)


class SlimNuScenes:
    """The subset of ``NuScenes.get`` the devkit eval uses, for the given samples only (streamed tables).

    sample: data[LIDAR_TOP], anns; sample_data: ego_pose_token; ego_pose; sample_annotation: + category_name.
    """

    def __init__(self, dataroot: Path, version: str, sample_tokens):
        d = Path(dataroot) / version
        keep = set(sample_tokens)
        self.tables = {"sample": {}, "sample_data": {}, "ego_pose": {}, "sample_annotation": {}}
        for r in iter_json_array(d / "sample.json"):
            if r["token"] in keep:
                self.tables["sample"][r["token"]] = {**r, "data": {}, "anns": []}
        sensor = {r["token"]: r["channel"] for r in iter_json_array(d / "sensor.json")}
        cs_channel = {r["token"]: sensor[r["sensor_token"]] for r in iter_json_array(d / "calibrated_sensor.json")}
        for r in iter_json_array(d / "sample_data.json"):
            if r["is_key_frame"] and r["sample_token"] in keep and cs_channel[r["calibrated_sensor_token"]] == "LIDAR_TOP":
                self.tables["sample_data"][r["token"]] = r
                self.tables["sample"][r["sample_token"]]["data"]["LIDAR_TOP"] = r["token"]
        poses = {r["ego_pose_token"] for r in self.tables["sample_data"].values()}
        for r in iter_json_array(d / "ego_pose.json"):
            if r["token"] in poses:
                self.tables["ego_pose"][r["token"]] = r
        cats = {r["token"]: r["name"] for r in iter_json_array(d / "category.json")}
        inst = {r["token"]: cats[r["category_token"]] for r in iter_json_array(d / "instance.json")}
        for r in iter_json_array(d / "sample_annotation.json"):
            if r["sample_token"] in keep:
                self.tables["sample_annotation"][r["token"]] = {**r, "category_name": inst[r["instance_token"]]}
                self.tables["sample"][r["sample_token"]]["anns"].append(r["token"])
        missing = keep - self.tables["sample"].keys()
        assert not missing, f"{len(missing)} samples not in {d}"

    def get(self, table: str, token: str) -> dict:
        return self.tables[table][token]


def to_eval_boxes(nusc: SlimNuScenes, data: dict, tokens: list[str], max_boxes: int) -> EvalBoxes:
    """Ego-frame boxes [x, y, z_centre, w_heading, l, h, yaw, vx, vy] -> global-frame DetectionBoxes."""
    out = EvalBoxes()
    for token in tokens:
        entry = data["results"][token]
        sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        q_e2g = Quaternion(pose["rotation"])
        rot, trans, yaw_e2g = q_e2g.rotation_matrix, np.asarray(pose["translation"]), q_e2g.yaw_pitch_roll[0]
        boxes = []
        for box, score, label in zip(entry["boxes"], entry["scores"], entry["labels"]):
            x, y, z, w, l, h, yaw, vx, vy = box
            boxes.append(DetectionBox(
                sample_token=token,
                translation=tuple(rot @ np.array([x, y, z]) + trans),
                size=(l, w, h),  # devkit wlh: width lateral, length along the heading
                rotation=tuple(Quaternion(axis=[0, 0, 1], radians=yaw + yaw_e2g).elements),
                velocity=tuple(rot[:2, :2] @ np.array([vx, vy])),
                detection_name=data["class_names"][label],
                detection_score=float(score),
                attribute_name="",
            ))
        out.add_boxes(token, sorted(boxes, key=lambda b: -b.detection_score)[:max_boxes])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataroot", type=Path, default=ROOT / "data/nuscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--max-boxes", type=int, default=500)
    p.add_argument("--recall", action="store_true", help="also per class and match distance the max recall "
                   "(share of GT ever matched, any score) and the precision at recall 0.3 / 0.5 -> metrics.json recall_diag")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gt_data, pred_data = json.loads(args.gt.read_text()), json.loads(args.pred.read_text())
    assert gt_data["class_names"] == pred_data["class_names"] == list(CLASS_RANGE), "class order mismatch"
    tokens = sorted(gt_data["results"])
    missing = set(tokens) - set(pred_data["results"])
    assert not missing, f"{len(missing)} GT samples have no predictions"
    nusc = SlimNuScenes(args.dataroot, args.version, tokens)

    cfg = DetectionConfig(class_range=CLASS_RANGE, dist_fcn="center_distance", dist_ths=list(DIST_THS),
                          dist_th_tp=DIST_TH_TP, min_recall=MIN_RECALL, min_precision=MIN_PRECISION,
                          max_boxes_per_sample=args.max_boxes, mean_ap_weight=5)
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, to_eval_boxes(nusc, gt_data, tokens, 10**6)), CLASS_RANGE)
    pred = filter_eval_boxes(nusc, add_center_dist(nusc, to_eval_boxes(nusc, pred_data, tokens, args.max_boxes)), CLASS_RANGE)

    per_class, recall_diag = {}, {}
    for name in CLASS_RANGE:
        n_gt = sum(b.detection_name == name for b in gt.all)
        if n_gt == 0:
            continue
        mds = {th: accumulate(gt, pred, name, cfg.dist_fcn_callable, th) for th in DIST_THS}
        aps = {th: calc_ap(m, MIN_RECALL, MIN_PRECISION) for th, m in mds.items()}
        md = accumulate(gt, pred, name, cfg.dist_fcn_callable, DIST_TH_TP)
        tps = {m: float("nan") if m in TP_EXCLUDED.get(name, ()) else float(calc_tp(md, MIN_RECALL, m)) for m in TP_USED}
        per_class[name] = {"n_gt": n_gt, "AP": float(np.mean(list(aps.values()))),
                           "AP_by_dist": {str(k): float(v) for k, v in aps.items()}, **tps}

    m_ap = float(np.mean([r["AP"] for r in per_class.values()]))
    tp = {m: float(np.nanmean([r[m] for r in per_class.values()])) for m in TP_USED}
    nds = (5 * m_ap + sum(1 - min(1.0, e) for e in tp.values())) / 10  # attribute term contributes 0
    report = {"gt": str(args.gt), "pred": str(args.pred), "samples": len(tokens),
              "protocol": "nuScenes devkit matching/AP/TP errors, teacher 7-class taxonomy, NDS without the attribute term",
              "mAP": m_ap, "NDS": nds, "NDS_aae0": nds + 0.1, "tp_errors": tp, "classes_in_mAP": list(per_class), "per_class": per_class,
              **({"recall_diag": recall_diag} if args.recall else {})}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))

    print(f"mAP {m_ap:.4f}   NDS {nds:.4f} (NDS_aae0 {nds + 0.1:.4f})   " + "   ".join(f"{m} {v:.4f}" for m, v in tp.items()))
    print(f"{'class':15s} {'n_gt':>5s} {'AP':>6s} {'@0.5':>6s} {'@1':>6s} {'@2':>6s} {'@4':>6s} {'ATE':>6s} {'ASE':>6s} {'AOE':>6s} {'AVE':>6s}")
    for name, r in per_class.items():
        a = r["AP_by_dist"]
        print(f"{name:15s} {r['n_gt']:5d} {r['AP']:6.3f} {a['0.5']:6.3f} {a['1.0']:6.3f} {a['2.0']:6.3f} {a['4.0']:6.3f} "
              + " ".join(f"{r[m]:6.3f}" for m in TP_USED))


if __name__ == "__main__":
    main()
