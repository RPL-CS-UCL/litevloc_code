#!/usr/bin/env bash

sessions=(
  "ucl_campus_20240904_0835"
  "ucl_campus_20241127_1716"
  "ucl_campus_20241128_1103"
  "ucl_campus_20241129_1144"
  "ucl_campus_20241129_1206"
  "ucl_campus_20241202_1415"
  "ucl_campus_20241202_1421"
  "ucl_campus_20241202_1425"
  "ucl_campus_20241202_1741"
  "ucl_campus_20241202_1746"
  "ucl_campus_20241204_1355"
  "ucl_campus_20241204_1359"
  "ucl_campus_20241204_1409"
  "ucl_campus_20241204_1418"
  "ucl_campus_20241204_1430"
  "ucl_campus_20241204_1438"
  "ucl_campus_20241204_1646"
  "ucl_campus_20241204_1649"
  "ucl_campus_20241204_1658"
  "ucl_campus_20241204_1700"
  "ucl_campus_20241204_1718"
  "ucl_campus_20241204_1802"
  "ucl_campus_20241205_1008"
  "ucl_campus_20241205_1017"
  "ucl_campus_20241221_1729"
  "ucl_campus_20241222_1242"
  "ucl_campus_20241222_1544"
  "ucl_campus_20241222_1637"
  "ucl_campus_20241223_1713"
  "ucl_campus_20241223_1733"
  "ucl_campus_20241223_1847"
)
scene_id=0
start_indice=0
length_segment=300

for session in "${sessions[@]}"; do
  echo "Processing Sessions: $session"
  python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_dataset/map_multisession/gendataset_from_files.py \
    --in_dir "/Rocket_ssd/dataset/data_litevloc/raw_vrs/ucl_campus/out_general_${session}" \
    --out_dir "/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus" \
    --scene_id $scene_id \
    --start_indice $start_indice \
    --split_length $length_segment \
    --kf_time_interval 3.0
done
