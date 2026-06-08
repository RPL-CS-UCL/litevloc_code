#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 <dataset_name>"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

DATASET_NAME="$1"

# Configurable paths
EGOBLUR_PATH="/Titan/code/robohike_ws/src/EgoBlur"
FACE_MODEL_PATH="/Rocket_ssd/image_matching_model_weights/ego_blur_face.jit"
LP_MODEL_PATH="/Rocket_ssd/image_matching_model_weights/ego_blur_lp.jit"
DATA_PATH="/Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/images"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: DATA_PATH '$DATA_PATH' does not exist."
    exit 2
fi

# Process all scenes
for SPLIT in "$DATA_PATH"/*; do
    [ -d "$SPLIT" ] || continue
    SPLIT_NAME=$(basename "$SPLIT")
    echo "Processing split: $SPLIT_NAME"

    for SCENE in database queries; do
        INPUT_DIR="$SPLIT/$SCENE"
        OUTPUT_DIR="$SPLIT/${SCENE}_blur"

        if [ ! -d "$INPUT_DIR" ]; then
            echo "Warning: '$INPUT_DIR' does not exist, skipping."
            continue
        fi

        echo "  Processing $SPLIT images..."
        mkdir -p "$OUTPUT_DIR"

        # Find and process all .jpg images
        find "$INPUT_DIR" -type f -iname "*.jpg" | while IFS= read -r INPUT_IMAGE; do
            FILENAME=$(basename "$INPUT_IMAGE")
            OUTPUT_IMAGE="$OUTPUT_DIR/$FILENAME"

            python "$EGOBLUR_PATH/script/demo_ego_blur.py" \
                --face_model_path "$FACE_MODEL_PATH" \
                --lp_model_path "$LP_MODEL_PATH" \
                --face_model_score_threshold 0.2 \
                --lp_model_score_threshold 0.5 \
                --input_image_path "$INPUT_IMAGE" \
                --output_image_path "$OUTPUT_IMAGE"
        done

        # Move original input to *_raw, move blurred output to original location
        mv "$INPUT_DIR" "$SPLIT/${SCENE}_raw"
        mv "$OUTPUT_DIR" "$INPUT_DIR"
    done
done
