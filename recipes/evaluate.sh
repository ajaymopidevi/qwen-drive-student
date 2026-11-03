#!/usr/bin/env bash
# Accuracy of the final chain.   EVAL_PY=<nuScenes env python> bash recipes/evaluate.sh [split, default val_quick]
# Writes outputs/eval/: detection metrics, map and occupancy mIoU against GT. Needs data/ (README, Data).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
source recipes/chain.sh
S=${1:-val_quick}
STRIDE=${STRIDE:-3}
O=outputs/eval
SPL=data/splits/v1.0-trainval.json
mkdir -p $O
python scripts/evaluation/eval_stage1.py --ckpt $STUDENT --trt-voxel $VOXEL --trt-refiner $REFINER --merged-pool --chain "$CHAIN" \
    --trt-student $STUDENT_ENGINE --images jpeg_gpu --packed data/packed/v1.0-trainval --splits $SPL --split $S --output $O/run
python scripts/evaluation/export_det_preds.py --preds $O/run/preds --output $O/pred.json --packed data/packed/v1.0-trainval --splits $SPL --split $S
python scripts/evaluation/export_det_preds.py --from-gt --output $O/gt.json --packed data/packed/v1.0-trainval --splits $SPL --split $S
python scripts/heading/dump_heading_crops.py pred --pred $O/pred.json --output $O/crops --workers 2
python scripts/heading/apply_heading_refiner.py --pred $O/pred.json --crops $O/crops --ckpt $HEADING --output $O/pred_heading.json \
    --gt $O/gt.json --keep-vel 2 --max-speed 2:1.0
$EVAL_PY scripts/evaluation/nusc_det_eval.py --version v1.0-trainval --gt $O/gt.json --pred $O/pred_heading.json --output $O/det.json
python scripts/evaluation/dump_map_logits.py --ckpt $STUDENT --label final --images jpeg_gpu --trt-voxel $VOXEL --trt-refiner $REFINER --merged-pool \
    --chain "$CHAIN" --trt-student $STUDENT_ENGINE --split $S --stride $STRIDE --output $O/dumps
python scripts/evaluation/eval_dense_gt.py --models final_SS --baseline final_SS --dumps $O/dumps --split $S --stride $STRIDE --output $O/dense.json
