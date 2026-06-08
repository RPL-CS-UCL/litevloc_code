#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
	echo "Error: DATASET_NAME is not specified."
	echo "Usage: ./run_finetune_rpe_test.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus_robot, mapfree, hkust_aria, ucl_campus_aria) <PATH_ESTIMATOR> <MODEL> (duster_calib_pretrain, master_calib_pretrain)"
	exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1
ESTIMATOR_PROJ_DIR=$2
MODEL=$3

export LITEVLOC_PROJ_DIR="/Titan/code/robohike_ws/src/litevloc"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval"

# Run Depth Generation
echo "Run Depth Generation"
rosrun litevloc run_benchmark_rpe_depth_generation.sh $DATASET_NAME 1 $MODEL

python $LITEVLOC_PROJ_DIR/python/benchmark_rpe/merge_pair_name.py --dataset_dir "$DATASET_PATH/finetune_$MODEL" --depth_suffix pdepth
python $LITEVLOC_PROJ_DIR/python/benchmark_rpe/merge_pair_name.py --dataset_dir "$DATASET_PATH/finetune_$MODEL" --depth_suffix gtdepth
python $LITEVLOC_PROJ_DIR/python/benchmark_rpe/merge_pair_name.py --dataset_dir "$DATASET_PATH/finetune_$MODEL" --depth_suffix m3ddepth

find "$DATASET_PATH"/test/ -name "*.pdepth.png" -delete
find "$DATASET_PATH"/test/ -name "*_pseudo.txt" -delete
cp -r "$DATASET_PATH"/finetune_"$MODEL"/pairs/s* "$DATASET_PATH"/train/

# Run Training
echo "Run Training"
cd $ESTIMATOR_PROJ_DIR
./test_lora/scripts/run_train_finetune_lora.sh "$DATASET_PATH" $DATASET_NAME
cd $LITEVLOC_PROJ_DIR

# Run RPE Evaluation
echo "Run RPE Evaluation"
bash $LITEVLOC_PROJ_DIR/scripts/run_benchmark_rpe_submission.sh $DATASET_NAME test
bash $LITEVLOC_PROJ_DIR/scripts/run_benchmark_rpe_evaluation.sh $DATASET_NAME test