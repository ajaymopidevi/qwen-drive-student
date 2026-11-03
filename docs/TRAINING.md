# Training lineage

The final model comes from a chain of runs, each starting from the previous checkpoint. The commands below are the ones
used, in order, with paths shortened. `RUN` names are free; each step reads the previous step's `runs/<name>/`.

The later trainers and engine builds were smoke-tested on this repository for a few steps, and the evaluation code was
checked frame by frame against the final model's outputs. The full chain has not been rerun from scratch here. A rerun will
not give identical weights, and engine builds vary (see Deployment).

Common variables:

```bash
SPL=data/splits/v1.0-trainval.json
PACKED=data/packed/v1.0-trainval
CACHE=cache/teacher/v1.0-trainval
ENG=<voxel engine from step 3>
E52=<occupancy refiner engine from step 3>
SKIP6=0,1,2,3,4,5
```

The steps follow the repository's folders. Steps 3 and 4 interleave: the voxel block in step 3 needs the head checkpoint from step 4b.

## 1. Dataset curation

The curated frames are already in `data/splits/v1.0-trainval.json`: `s1_C` (2,000 frames) and `s1_C4k` (4,000), with `s1_C` a subset of
`s1_C4k`. You only need the commands below to redo the selection. They have not been rerun on this repository, and they overwrite that
file, so keep a copy.

```bash
# rank nuScenes scenes for selection
python scripts/dataset/select_scenes.py
# thin the keyframes before selection
python scripts/dataset/select_frames.py
# DINOv2 embeddings of the candidate frames
python scripts/dataset/embed_frames.py --output data/splits/v1.0-trainval_dinov2_kept.npz
# pick frames by rare-class gain + k-center
python scripts/dataset/build_arms.py --budget 2000                      # s1_A..s1_D; s1_C is the one used
# grow a selection, keeping its frames
python scripts/dataset/extend_arm.py --base s1_C --budget 4000 --name s1_C4k
```

`extend_arm.py` keeps the earlier frames, so their teacher outputs are reused. For another 2K, run it again with `--base s1_C4k --budget 6000
--name s1_C6k`. Only the 2K and 4K selections were trained.

## 2. Backbone distillation

```bash
# run the teacher on the frames and cache its taps and head outputs
python scripts/backbone/cache_teacher.py --frames $PACKED --output $CACHE --splits $SPL --split s1_C4k --phase vlm --tap-format vitW+llmInt8
# convert cached taps to the compact training format
python scripts/backbone/convert_tap_store.py --split val_quick --format vitW+llmInt8
# per-channel mean and std of the taps, to normalise targets
python scripts/backbone/tap_stats.py --cache $CACHE --splits $SPL --split s1_CB
# foreground class maps for the student loss
python scripts/backbone/precompute_fg.py --split s1_C4k

# train the student to regress the taps, with losses in the head space
python scripts/backbone/train_stage1.py --run-name student --epochs 12 --images jpeg --packed $PACKED --cache $CACHE --splits $SPL \
    --train-split s1_C4k --val-split val_quick --tap-stats $CACHE/tap_stats_s1_CB.npz \
    --taps-dir taps_vitW+llmInt8 --val-taps-dir taps_vitW+llmInt8 \
    --fg-root cache/fg/v1.0-trainval --fg-alpha 1 --depth-kd 1 --head-space 1 --vit-raw-weight 0
```

## 3. Head distillation

### Voxel block

```bash
# rank-64 block at full height
# distill_voxel_block: feature KD on matched inputs + prediction KD through the frozen head
python scripts/head/distill_voxel_block.py --run-name block_r64_pca --arch teacher_shaped --rank 64 --init teacher_pca --lr 2e-4 --epochs 5 --resume
python scripts/head/distill_voxel_block.py --run-name block_r64 --init-block runs/block_r64_pca/best.pt --lr 1e-4 --epochs 3 \
    --stop-after 0 --checkpoint-layers --resume --pred-weight 0.2 --ckpt-head
# trt_voxel_block: export the block to ONNX, then build its TensorRT engine
python scripts/head/trt_voxel_block.py --phase export --anchor-blocks runs/block_r64/best.pt
python scripts/head/trt_voxel_block.py --phase build --only fp16 --anchor-blocks runs/block_r64/best.pt

# half height
python scripts/head/distill_voxel_block.py --run-name block_z8 --init-block runs/block_r64/best.pt --zhalf --lr 1e-4 --epochs 3 \
    --checkpoint-layers --resume --pred-weight 0.2 --ckpt-head

# warped grid (after step 4b gives a head checkpoint)
# warp_grid: build the block engine for the warped grid
python -m qdistill.warp_grid block-engine --b 25.6 --out weights/engines/block_warp150_fp16.engine
python scripts/head/distill_voxel_block.py --run-name block_z8_warp --init-block runs/block_z8/best.pt --z8-compress stride6 \
    --warp-grid 25.6 --warp-head runs/ft_warp/final.pt --ring-weight 4 --pred-weight 0.5 --ckpt-head --checkpoint-layers \
    --lr 1e-4 --epochs 2 --save-every 250 --resume
# zhalf: build the voxel block engine
python -m qdistill.zhalf build2 --ckpt runs/block_z8_warp/best.pt --out weights/engines/block_z8_fp16.engine --grid 150
```

### Occupancy refiner

```bash
# distill_refiner: cache refiner inputs, train from the teacher refiner, build the engine
python scripts/head/distill_refiner.py --phase cache --cache-frames 2000
python scripts/head/distill_refiner.py --phase train --config c32_64_b1 --occ-weight 1 --run-name refiner_w1 --resume
python scripts/head/distill_refiner.py --phase build --run-name refiner_w1
# then the 2D refiner, distilled from the refiner above
COMMON="--config occ2d_w128 --init random --occ-weight 2 --feat-occ-weight 2 --refiner-engine <refiner_w1 engine>"
python scripts/head/distill_refiner.py --phase train $COMMON --run-name occ2d --epochs 12 --resume
python scripts/head/distill_refiner.py --phase build $COMMON --run-name occ2d
```

## 4. Sparse queries

### Encoder and dense heads

Each fine-tune starts from the one before it. Common flags:
`FT="--voxel-engine $ENG --refiner-engine $E52 --w-det 1 --occ-weight 4 --skip-ca $SKIP6 --lr 1e-4 --train-stride 1 --w-gt-map 1 --w-gt-occ 1 --enc-precision fp32 --val-frames 10 --log-every 250"`

```bash
# train_enc_recover: fine-tune the BEV encoder and dense heads in deployment numerics (KD + GT map and occupancy)

# a. no image reads in the encoder
python scripts/sparse_queries/train_enc_recover.py --run-name ft_noca --voxel-engine $ENG --refiner-engine $E52 --w-det 0 --w-occ 10 --occ-weight 4 \
    --skip-ca $SKIP6 --steps 2000 --lr 1e-4
python scripts/sparse_queries/train_enc_recover.py --run-name ft_fp32 $FT --w-occ 10 --steps 3000 --init-enc runs/ft_noca/final.pt
python scripts/sparse_queries/train_enc_recover.py --run-name ft_numerics $FT --w-occ 10 --steps 3000 --init-enc runs/ft_fp32/final.pt \
    --front-trt <vit_neck + depth_net engines> --train-heads map_seg,uvtr_occ_fuse,occ_pred_head --heads-lr 2e-5

# b. warped grid
python scripts/sparse_queries/train_enc_recover.py --run-name ft_warp $FT --w-occ 10 --steps 3000 --init-enc runs/ft_numerics/final.pt \
    --init-heads runs/ft_numerics/final.pt --front-trt <engines> --train-heads map_seg,uvtr_occ_fuse,occ_pred_head --heads-lr 2e-5 \
    --warp-grid weights/engines/block_warp150_fp16.engine --warp-b 25.6

# c. voxel block, then trainable UVTR projections (the released fine_tune.pt)
python scripts/sparse_queries/train_enc_recover.py --run-name ft_z8 $FT --w-occ 15 --steps 4500 --init-enc runs/ft_warp/final.pt \
    --init-heads runs/ft_warp/final.pt --front-trt <engines> --train-heads map_seg,uvtr_occ_fuse,occ_pred_head --heads-lr 2e-5 \
    --warp-grid weights/engines/block_warp150_fp16.engine --warp-b 25.6 --warp-z8 weights/engines/block_z8_fp16.engine
python scripts/sparse_queries/train_enc_recover.py --run-name fine_tune $FT --w-occ 15 --steps 4500 --init-enc runs/ft_z8/final.pt \
    --init-heads runs/ft_z8/final.pt --front-trt <engines> \
    --train-heads map_seg,uvtr_occ_fuse,occ_pred_head,uvtr_query_proj,uvtr_occ_proj --heads-lr 2e-5 \
    --warp-grid weights/engines/block_warp150_fp16.engine --warp-b 25.6 --warp-z8 weights/engines/block_z8_fp16.engine
```

### Query camera sampler

```bash
# train_det_sampling: refit the query camera sampler on each dense checkpoint (GT + KD)
python scripts/sparse_queries/train_det_sampling.py --run-name smp_0 --dense noca:runs/ft_noca/final.pt --voxel-engine $ENG --refiner-engine $E52 --steps 4000 --w-gt 0.5
python scripts/sparse_queries/train_det_sampling.py --run-name smp_1 --dense noca:runs/ft_fp32/final.pt --enc-precision fp32 --init-sampler runs/smp_0/final.pt \
    --w-gt 0.5 --voxel-engine $ENG --refiner-engine $E52 --steps 2000
python scripts/sparse_queries/train_det_sampling.py --run-name smp_2 --dense noca:runs/ft_numerics/final.pt --enc-precision fp32 --init-sampler runs/smp_1/final.pt \
    --w-gt 0.5 --voxel-engine $ENG --refiner-engine $E52 --front-trt <engines> --steps 2000
python scripts/sparse_queries/train_det_sampling.py --run-name smp_3 --dense noca:runs/ft_warp/final.pt --enc-precision fp32 --init-sampler runs/smp_2/final.pt \
    --w-gt 0.5 --voxel-engine $ENG --refiner-engine $E52 --front-trt <engines> --steps 2000 --warp-offsets
python scripts/sparse_queries/train_det_sampling.py --run-name smp_4 --dense noca:runs/ft_z8/final.pt --enc-precision fp32 --init-sampler runs/smp_3/final.pt \
    --w-gt 0.5 --voxel-engine $ENG --refiner-engine $E52 --front-trt <engines> --steps 2000 --warp-offsets
python scripts/sparse_queries/train_det_sampling.py --run-name sampler --dense noca:runs/fine_tune/final.pt --enc-precision fp32 --init-sampler runs/smp_4/final.pt \
    --w-gt 0.5 --voxel-engine $ENG --refiner-engine $E52 --front-trt <engines> --steps 2000 --warp-offsets
```

## 5. Heading refiner

```bash
# camera crops of pedestrian and bicycle boxes
python scripts/heading/dump_heading_crops.py gt --output outputs/heading_crops/gt --frame-stride 1 --workers 2
# train the heading refiner on the GT crops
python scripts/heading/train_heading_refiner.py --crops outputs/heading_crops/gt --run-name heading_refiner
# export and build its TensorRT engine
python scripts/heading/trt_heading_refiner.py --ckpt runs/heading_refiner/best.pt
```

## 6. Deployment

`CHAIN` below is the part of `recipes/chain.sh` that exists at that step.

```bash
# BEV encoder engine: export, then build
python scripts/deployment/trt_bev_encoder.py --phase export --chain "$AB" --enc-fp32 weights/fine_tune.pt --output outputs/engines
python scripts/deployment/trt_bev_encoder.py --phase build --output outputs/engines
# small head modules as engines
python scripts/deployment/trt_modules.py --chain "$AB" --heads-ckpt weights/fine_tune.pt --names <name> --output outputs/engines/modules
# fused occupancy + map engine: export, then build
python scripts/deployment/trt_dense_heads.py --phase export --crop grid --chain "$CHAIN" --heads-ckpt weights/fine_tune.pt --frames 4 --output outputs/engines
python scripts/deployment/trt_dense_heads.py --phase build --output outputs/engines

# student engine with the linear layers folded: export, then build
python scripts/deployment/trt_student_fpn.py --fold --phase export --output outputs/student
python scripts/deployment/trt_student_fpn.py --fold --phase build --output outputs/student
# fused student front (neck + depth network), exported for step 7
python scripts/deployment/trt_student_fpn.py --front --phase export --output outputs/student
```

`AB` is `$SKIP,enc_ckpt:<fine_tune>,heads_ckpt:<fine_tune>,warp_grid:25.6:<block>:<fine_tune>,zhalf2:<z8 block>:fold`.

The CUDA head (graph decoder, graph head, merged pooling in `src/qdistill/csrc/`) compiles on first use, with `source env.sh`.

TensorRT's attention fusion can differ between builds of the same ONNX. An unfused student engine is much slower.
`build_student_engine.sh` checks the profile and rebuilds up to three times. Check the same way after any other student build.

## 7. Backbone quantization

```bash
# calibrate activation ranges
python scripts/quantization/quant_student.py --phase calib --output outputs/student
# share scales per block
python scripts/quantization/quant_student.py --phase share --method hybrid --output outputs/student
# insert Q/DQ, build the INT8 engine, rebuild until attention is fused
bash recipes/build_student_engine.sh
```

## 8. Evaluation

`recipes/evaluate.sh` scores a chain. `scripts/evaluation/nusc_det_bootstrap.py` and `eval_dense_gt.py` give the paired scene bootstrap
behind the accuracy gate.

## Profiling

`recipes/latency.sh` runs `scripts/profiling/e2e_latency.py`: median time per stage over val_quick. `trt_student_fpn.py --phase profile` gives
per-layer times for the student engine, and `recipes/build_student_engine.sh` uses it to check attention fusion.
