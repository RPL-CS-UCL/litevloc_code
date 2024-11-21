#!/bin/bash

# Dense matching: 
#   roma tiny-roma duster master
# Semi-dense matching:
#   loftr eloftr matchformer xfeat-star
# Sparse matching:
#   sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_mf_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export OUT_DIR="/Rocket_ssd/dataset/data_litevloc/$DATASET_NAME/map_free_eval/results_mf"
export MODELS="roma tiny-roma duster master loftr eloftr matchformer xfeat-star sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat"

python $PROJECT_PATH/python/benchmark_map_free/submission.py --config $CONFIG_FILE --models $MODELS --pose_solver pnp \
  --out_dir $OUT_DIR \
  --split test --debug