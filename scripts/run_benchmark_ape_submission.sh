#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_ape_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export OUT_DIR="/Rocket_ssd/dataset/data_litevloc/$DATASET_NAME/map_free_eval/results_ape"
export MODELS="master"

# Run the Python script
python $PROJECT_PATH/python/benchmark_ape/submission.py --config $CONFIG_FILE --models $MODELS --out_dir $OUT_DIR --split test --debug
