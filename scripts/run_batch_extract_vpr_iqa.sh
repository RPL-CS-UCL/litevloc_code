#!/bin/bash

# Define path and number of submaps
PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00000_atrium_aria_data"
NUM_SUBMAP=5

# Loop over each submap index
for ((i=0; i<$NUM_SUBMAP; i++ ))
do
    echo "Processing submap index: $i"
    rosrun litevloc extract_vpr_descriptors.py \
      --map_path "$PATH_SUBMAP/$i" \
      --method cosplace \
      --backbone ResNet18 \
      --descriptors_dimension 256 \
      --num_preds_to_save 3 \
      --image_size 512 288 \
      --device cuda \
      --save_descriptors
done

for ((i=0; i<$NUM_SUBMAP; i++ ))
do
    echo "Processing submap index: $i"
    rosrun litevloc extract_iqa.py \
      --map_path "$PATH_SUBMAP/$i" \
      --metric musiq \
      --device cuda
done