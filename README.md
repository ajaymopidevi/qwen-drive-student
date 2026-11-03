# Qwen-Drive Student

Distilling a driving VLM's 3D perception into a small camera-only model that runs on one RTX 4060 (8 GB).

**[Project page](https://ajaymopidevi.github.io/qwen-drive-student/)**: method, interactive 3D viewer and results.

[Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0) produces 3D detection, BEV map segmentation and 3D occupancy
from six cameras, on top of a 4-billion-parameter vision-language model. This repository replaces the VLM with a 24M-parameter
student and rebuilds the perception head block by block. Each replacement had to pass a pre-set accuracy gate before the
next one started.

## Results

val_quick: 30 held-out nuScenes scenes, 1,201 frames. Map and occupancy use every third frame.

| Milestone | Det mAP | NDS | Map mIoU | Occ mIoU | Latency |
|---|---:|---:|---:|---:|---:|
| Teacher | 0.467 | 0.445 | 0.584 | 0.292 | ~4,000 ms |
| Backbone distillation, 2K frames | 0.411 | 0.394 | | | |
| Backbone distillation, 4K frames | 0.422 | 0.407 | | | ~1,250 ms |
| Head distillation and TensorRT engines | 0.419 | 0.404 | 0.563 | 0.283 | 416 ms |
| Sparse queries, heading refiner and deployment fine-tune | 0.433 | 0.412 | 0.592 | 0.286 | 184 ms |
| Warped BEV grid, fused heads, exact folds | 0.436 | 0.422 | 0.599 | 0.283 | 118 ms |
| INT8 backbone | 0.434 | 0.420 | 0.597 | 0.283 | 116 ms |
| **Glue removal (final model)** | **0.434** | **0.420** | **0.597** | **0.283** | **105 ms** |

- The 1,250 ms is an estimate: the measured 1,223 ms head plus the student. Earlier rows come from different sessions, which vary by a few ms.
- Final latency is the median per six-camera frame on an RTX 4060 Laptop GPU: student 20.9 ms, head 78.2 ms, heading refiner 4.9 ms.
  It excludes JPEG decode, which adds about 22 ms (126.3 ms in total).
- Detection uses the teacher's 7-class taxonomy and its NDS convention, without the attribute term. Map is scored against
  nuScenes map-expansion rasters and occupancy against Occ3D, on the camera mask.
- Full nuScenes validation (6,019 frames; map and occupancy on every third frame): the final model scores mAP 0.405, NDS 0.402,
  map 0.596 and occupancy 0.282, against 0.439, 0.428, 0.601 and 0.287 for the teacher (92%, 94%, 99% and 98%).

Only the final model's weights are released. The numbers and the accuracy gate are in [results/metrics.json](results/metrics.json).

## How it works

1. **Dataset curation.** Scenes and frames chosen for diversity and rare classes: 2K frames, then 4K, nested so teacher outputs are reused.
2. **Backbone distillation.** An HGNetV2-B4 student regresses the two features the teacher's head reads, with the loss in
   the head's own projected space plus a depth distribution KL.
3. **Head distillation.** A low-rank voxel block at half height, a 2D occupancy refiner and a warped BEV grid: 0.512 m cells
   within 25.6 m of the car and 1.024 m beyond.
4. **Sparse queries.** The BEV encoder stops reading images. Detection queries sample image levels through a trained camera sampler.
5. **Heading refiner.** A small crop classifier that flips the yaw of pedestrians and bicycles where it disagrees.
6. **Deployment.** TensorRT engines, a fused dense-head engine, exact folds of linear layers and CUDA head kernels, with the neck
   and depth network fused into the student and needless precision casts removed.
7. **Backbone quantization.** An INT8 backbone with one activation scale per block.
8. **Evaluation.** A paired scene bootstrap with an accuracy floor gates every step.

## Repository layout

```
scripts/dataset/         nuScenes packing, ground truth, frame selection
scripts/backbone/        teacher cache, student training
scripts/head/            voxel block and occupancy refiner distillation and engines
scripts/sparse_queries/  encoder fine-tune, query camera sampler
scripts/heading/         heading refiner
scripts/deployment/      TensorRT engines
scripts/quantization/    INT8 student
scripts/evaluation/      detection, map and occupancy scoring, bootstrap
scripts/profiling/       end-to-end latency per stage
src/qdistill/            models, TensorRT runners, CUDA kernels; head_chain.py assembles the deployed head
recipes/                 chain.sh (the final chain), evaluate.sh, latency.sh, build_student_engine.sh
results/                 reported numbers
weights/                 manifest.json: released files and checksums
docs/                    TRAINING.md: every step, in order; index.html and assets/: the project page
```

## Setup

Python 3.12, CUDA 12.8, an NVIDIA GPU with 8 GB or more.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/QwenLM/Qwen-Drive-1.0 third_party/Qwen-Drive-1.0
git -C third_party/Qwen-Drive-1.0 checkout 28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0
pip install -e third_party/Qwen-Drive-1.0 --no-deps --no-build-isolation
pip install -e . --no-deps

# the nuScenes devkit needs its own environment
python -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt
```

Edit `env.sh` if your CUDA toolkit is elsewhere or your GPU is not an RTX 4060, and run `source env.sh` before running anything. It also sets `PYTHONPATH`.

### Weights

```bash
python scripts/download_weights.py                 # students, engines, ONNX, calibration: about 660 MB
```

The teacher's weights are needed too, in `weights/Qwen-Drive-1.0-4B`: follow the Qwen-Drive-1.0 instructions.

The released TensorRT engines were built with `tensorrt-cu12==10.13.3.9` for an RTX 4060 and may not load elsewhere. If so, rebuild
them: [docs/TRAINING.md](docs/TRAINING.md), Deployment.

### Data

Download nuScenes trainval with the map expansion to `data/nuscenes` and the Occ3D labels to `data/occ3D_nuScenes`, then:

```bash
python scripts/dataset/pack_nuscenes.py --help
python scripts/dataset/extract_ego_poses.py
.venv-eval/bin/python scripts/dataset/map_gt.py --split val_quick --stride 3
.venv-eval/bin/python scripts/dataset/map_gt.py --split val_rest --stride 9
```

`data/splits/v1.0-trainval.json` already holds the splits and the selected training frames.

## Reproduce the results

```bash
EVAL_PY=.venv-eval/bin/python bash recipes/evaluate.sh    # outputs/eval: det.json/metrics.json, dense.json
bash recipes/latency.sh                                   # outputs/latency
```

## Training

[docs/TRAINING.md](docs/TRAINING.md) lists every training step in order. A full rerun takes days and needs the teacher cache.

## Acknowledgements and license

Code is under the Apache-2.0 license (see LICENSE). This project is independent work, not affiliated with the Qwen-Drive authors,
nuScenes or Motional. It builds on [Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0). nuScenes is under CC BY-NC-SA 4.0;
derived weights and outputs are subject to its terms. Occupancy labels are from Occ3D.
