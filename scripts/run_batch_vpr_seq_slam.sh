#!/bin/bash

# Define path and number of submaps
PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/hkust/s00000"
NUM_SUBMAP=6

# Loop over each submap index
for ((i=1; i<$NUM_SUBMAP; i++ ))
do
    echo "Processing submap index: $i"

    # Run descriptor extraction
    rosrun litevloc vpr_sequence_matching.py \
      --db_map_path "$PATH_SUBMAP/out_map0" \
      --query_map_path "$PATH_SUBMAP/out_map$i"
done
