# The final chain: sourced by the other recipes.
W=${WEIGHTS:-weights}
E=$W/engines
SKIP="skip_ca:0,skip_ca:1,skip_ca:2,skip_ca:3,skip_ca:4,skip_ca:5"
HEADS_OK=$W/fine_tune.pt
ENGS="vit_neck+depth_net+occ_pred_head+map_seg"
CHAIN="last_out,$SKIP,enc_ckpt:$HEADS_OK,heads_ckpt:$HEADS_OK,warp_grid:25.6:$E/block_warp150_fp16.engine:$HEADS_OK,warp_offsets"
CHAIN="$CHAIN,enc_trt:$E/encoder_fp16.engine,mod_trt:$E/modules:$ENGS,heads_fp16:$HEADS_OK:uvtr_occ_fuse,fast_geom"
CHAIN="$CHAIN,det_sample:$W/sampler.pt,zhalf2:$E/block_z8_fp16.engine:fold,dense_trt:$E/dense_heads_fp16.engine,pool_fold,vol_fp16"
STUDENT=$W/student.pt
STUDENT_ENGINE=$E/student_front_int8.engine
VOXEL=$E/voxel_r64_fp16.engine
REFINER=$E/occ_refiner_fp16.engine
HEADING=$W/heading_refiner.pt
EVAL_PY=${EVAL_PY:-python}          # interpreter of the nuScenes environment
