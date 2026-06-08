#!/usr/bin/env bash

sessions=(
  "ucl_campus_20241128_1103-reference"
  "ucl_campus_20241129_1144-reference"
  "ucl_campus_20241204_1430-reference"
  "ucl_campus_20241204_1438-reference"
  "ucl_campus_20241222_1242-reference"
  "ucl_campus_20240904_0835-query"
  "ucl_campus_20241202_1741-query"
  "ucl_campus_20241202_1746-query"
  "ucl_campus_20241202_1415-query"
  "ucl_campus_20241204_1418-query"
  "ucl_campus_20241202_1421-query"
  "ucl_campus_20241204_1802-query"
  "ucl_campus_20241205_1008-query"
  "ucl_campus_20241202_1425-query"
  "ucl_campus_20241204_1355-query"
  "ucl_campus_20241204_1646-query"
  "ucl_campus_20241204_1649-query"
  "ucl_campus_20241204_1359-query"
  "ucl_campus_20241204_1409-query"
  "ucl_campus_20241205_1017-query"
  "ucl_campus_20241204_1718-query"
  "ucl_campus_20241127_1716-query"
  "ucl_campus_20241129_1206-query"
  "ucl_campus_20241204_1658-query"
  "ucl_campus_20241204_1700-query"
  "ucl_campus_20241221_1729-query"
  "ucl_campus_20241222_1544-query"
  "ucl_campus_20241222_1637-query"
  "ucl_campus_20241223_1713-query"
  "ucl_campus_20241223_1733-query"
  "ucl_campus_20241223_1847-query"
)
scene_id=0
start_indice=0
length_segment=300

for session_type in "${sessions[@]}"; do
  echo "Processing Sessions: $session_type"
  session=$(echo $session_type | cut -d'-' -f1)
  type=$(echo $session_type | cut -d'-' -f2)
  if [ "$type" == "reference" ]; then
    python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_dataset/map_multisession/gendataset_from_files.py \
      --out_type vpr \
      --in_dir "/Rocket_ssd/dataset/data_litevloc/raw_vrs/ucl_campus/out_general_${session}" \
      --out_dir "/Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/reference" \
      --scene_id $scene_id \
      --start_indice $start_indice \
      --split_length 1000000 \
      --kf_time_interval 3.0
  else
    python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_dataset/map_multisession/gendataset_from_files.py \
      --out_type vpr \
      --in_dir "/Rocket_ssd/dataset/data_litevloc/raw_vrs/ucl_campus/out_general_${session}" \
      --out_dir "/Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/query" \
      --scene_id $scene_id \
      --start_indice $start_indice \
      --split_length $length_segment \
      --kf_time_interval 3.0
  fi
done
