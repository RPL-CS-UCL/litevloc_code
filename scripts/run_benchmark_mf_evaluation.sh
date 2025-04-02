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
  echo "Usage: ./run_benchmark_mf_evaluation.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/"
export MODELS=(
  "roma"
  "tiny-roma"
  "duster"
  "master"
  "loftr"
  "eloftr"
  "matchformer"
  "xfeat-star"
  "sift-lg"
  "superpoint-lg"
  "gim-lg"
  "xfeat-lg"
  "sift-nn"
  "orb-nn"
  "gim-dkm"
  "xfeat"
)

for model in "${MODELS[@]}"
do
  echo "Evaluate pose_estimation methods: $model"
  python $PROJECT_PATH/python/benchmark_map_free/evaluation.py \
    --submission_path $DATASET_PATH/results_mf/"$model"_pnp/submission.zip \
    --dataset_path $DATASET_PATH \
    --split test \
    --log error
  echo ""
done
