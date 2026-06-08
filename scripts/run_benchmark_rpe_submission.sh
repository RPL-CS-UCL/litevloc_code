#!/bin/bash

# Configuration
NUM_PARALLEL=1  # Set desired parallelism level here

# Check if DATASET_NAME is provided
if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Error: DATASET_NAME is not specified."
	echo "Usage: ./run_benchmark_rpe_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus_aria, mapfree, 360loc_aria, 360loc_phone, 360loc_vehicle) 
	<SPLIT> (train, val, test)"
	exit 1
fi

# Set the DATASET_NAME variable from the arguments
DATASET_NAME=$1
SPLIT=$2

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export OUT_DIR="$DATASET_PATH/$DATASET_NAME/map_free_eval/results_rpe"
export N_QUERY=10

# Model configuration
MODELS=(
	# "hloc_superpoint_splg"
	# "hloc_disk_dilg"
	# "vpr_cosplace_resnet18_256"
	# "vpr_netvlad_resnet18_4096"
	# "reloc3r"
	"duster_nocalib_pretrain"
	"duster_calib_pretrain"
	"master_nocalib_pretrain"
	"master_calib_pretrain"
)

# Processing function for a model
process_model() {
	local MODEL="$1"
	local top_k="$2"
	echo "Processing model: $MODEL"
	echo "Loading dataset from $DATASET_PATH"
	python "$PROJECT_PATH/python/benchmark_rpe/submission.py" \
		--config "$CONFIG_FILE" \
		--models "$MODEL" \
		--out_dir "$OUT_DIR" \
		--n_query "$N_QUERY" \
		--top_k "$top_k" \
		--split "$SPLIT" \
		--crop_image_to_database \
		--debug # --viz
	echo ""
	sleep 3
}

# Export required variables and functions
export -f process_model
export PROJECT_PATH DATASET_PATH CONFIG_FILE OUT_DIR N_QUERY SPLIT

# Main processing loop
# for TOP_K in $(seq 2 3 17) $(seq 20 10 50); do
for TOP_K in $(seq 2 3 17); do
# for TOP_K in $(seq 30 3 30); do # for test
	export TOP_K
	echo "Processing with TOP_K: $TOP_K"
	printf "%s\n" "${MODELS[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'process_model "$@" "$TOP_K"' _ {}
	# Unzip files
	for MODEL in "${MODELS[@]}"; do
		mkdir -p "$OUT_DIR/$MODEL/submission_$TOP_K"
		if [ -f "$OUT_DIR/$MODEL/submission_$TOP_K.zip" ]; then
			unzip -o "$OUT_DIR/$MODEL/submission_$TOP_K.zip" -d "$OUT_DIR/$MODEL/submission_$TOP_K"
		else
			echo "Error: submission_$TOP_K.zip not found for model $MODEL"
		fi
	done
done
