#!/usr/bin/env bash
# Latency of the final chain: bash recipes/latency.sh   -> outputs/latency/latency_*.json
# Reported latency = median total - median decode.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
source recipes/chain.sh
python scripts/profiling/e2e_latency.py --variant trt_all --merged-pool --refiner-engine $REFINER --voxel-engine $VOXEL --heading $HEADING \
    --heading-engine $E/heading_refiner_fp16_b64.engine --graph-decoder --graph-head --fast-post --sync-trim --side-stream \
    --student-engine $STUDENT_ENGINE --chain "$CHAIN" --ckpt $STUDENT --output outputs/latency
python - <<'PY'
import glob, json
e = json.load(open(sorted(glob.glob("outputs/latency/latency_*.json"))[0]))["end_to_end"]
print({k: v["median_ms"] for k, v in e.items()}, "latency", round(e["total"]["median_ms"] - e["decode"]["median_ms"], 1), "ms")
PY
