#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_evaluation.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree, hkust_aria, 360loc_aria, 360loc_phone, 360loc_vehicle)
  <SPLIT> (train, val, test)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1
SPLIT=$2

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export N_QUERY=10
export EVAL_CONFIGS=("config_1_10") # Optional: config_005_5, config_025_5, config_05_10, config_1_10, config_2_20

export MODELS=(
	"hloc_disk_dilg"
  "hloc_superpoint_splg"
	"vpr_cosplace_resnet18_256"
	"vpr_netvlad_resnet18_4096"
  "reloc3r"
  "duster_nocalib_pretrain"
  "duster_calib_pretrain"
  "master_nocalib_pretrain"
  "master_calib_pretrain"
)

for TOP_K in $(seq 2 3 17) $(seq 20 10 50); do
  echo "Processing with TOP_K: $TOP_K"
  for EVAL_CONFIG in "${EVAL_CONFIGS[@]}"
  do
    for model in "${MODELS[@]}"
    do
      echo "Evaluate pose_estimation methods: $model"   
      python $PROJECT_PATH/python/benchmark_rpe/evaluation.py \
        --config $CONFIG_FILE \
        --submission_path $DATASET_PATH/$DATASET_NAME/map_free_eval/results_rpe/$model/submission_$TOP_K.zip \
        --dataset_path $DATASET_PATH/$DATASET_NAME/map_free_eval \
        --n_query $N_QUERY \
        --split $SPLIT \
        --eval_config $EVAL_CONFIG \
        --log error
      echo ""
    done
  done
done