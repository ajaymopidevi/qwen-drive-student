"""Export saved predictions to the JSON the nuScenes evaluation reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qdistill import metrics as M
from qdistill.data import split_tokens
from qwen_drive_perception.configuration_perception import DET_CLASS_NAMES
from qwen_drive_perception.dataset import PerceptionFrame

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--preds", type=Path, help="dir of <token>.npz with boxes/scores/labels (lidar frame)")
    src.add_argument("--from-gt", action="store_true", help="export packed GT (>=1 point, mapped label) as score-1 predictions")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="mini_val")
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits/v1.0-mini.json")
    p.add_argument("--packed", type=Path, default=ROOT / "data/packed/v1.0-mini")
    p.add_argument("--stride", type=int, default=1, help="every n-th token of --split (as eval_stage1 --stride)")
    args = p.parse_args()

    out = {}
    for token in split_tokens(args.splits, args.split)[:: args.stride]:
        frame = PerceptionFrame(args.packed / token)
        if args.from_gt:
            boxes, labels = M.filter_gt(frame.gt)
            scores = np.ones(len(boxes))
        else:
            pred = dict(np.load(args.preds / f"{token}.npz"))
            boxes, labels = M.pred_to_ego(pred, frame.lidar2ego, score_thr=-np.inf)
            scores = np.sort(pred["scores"])[::-1]  # pred_to_ego returns boxes sorted by descending score
        out[token] = {"boxes": boxes.tolist(), "scores": scores.tolist(), "labels": labels.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"class_names": list(DET_CLASS_NAMES), "results": out}))
    print(f"{len(out)} samples, {sum(len(v['scores']) for v in out.values())} boxes -> {args.output}")


if __name__ == "__main__":
    main()
