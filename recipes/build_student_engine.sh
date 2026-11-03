#!/usr/bin/env bash
# Rebuild the INT8 student engine from weights/onnx/student_front.onnx. TensorRT's attention fusion varies between builds
# of the same ONNX, so build until the profile shows fused attention (up to 3 builds).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
W=${WEIGHTS:-weights}
O=outputs/student
F=$O/student_front_int8.engine
Q=$O/student_front_qdq.onnx
mkdir -p $O
cp $W/calib.json $O/calib.json
python scripts/quantization/quant_student.py --phase qdq --method share_block --onnx-in $W/onnx/student_front.onnx --onnx-out $Q --output $O
for try in 1 2 3; do
  python scripts/quantization/quant_student.py --phase build --onnx-out $Q --engine $F
  python scripts/deployment/trt_student_fpn.py --phase profile --profile-engine $F --output $O/prof
  python - "$F" "$O" <<'PY' && exit 0
import json, sys
t = json.load(open(f"{sys.argv[2]}/prof/student_profile.json"))[sys.argv[1]]["top_layers_ms"]
raise SystemExit(0 if any("_gemm_mha" in k for k in t) and not any("MaxrSubExp" in k for k in t) else 1)
PY
  echo "attention not fused (try $try), rebuilding"; rm -f $F
done
echo "attention not fused after 3 builds"; exit 1
