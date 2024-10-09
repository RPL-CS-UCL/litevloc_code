export PYTHON_PROJECT_PATH=/Titan/code/robohike_ws/src/pycpptools/
export DATA_PATH=/Rocket_ssd/dataset/data_litevloc/matterport3d

export SCENES=("17DRP5sb8fy" "17DRP5sb8fy" "17DRP5sb8fy" "EDJbREhghzL" "EDJbREhghzL" "EDJbREhghzL" "B6ByNegPMK" "B6ByNegPMK" "B6ByNegPMK")
export SEQ_NAMES=("17DRP5sb8fy_seq0" "17DRP5sb8fy_seq1" "17DRP5sb8fy_seq2" "EDJbREhghzL_seq0" "EDJbREhghzL_seq1" "EDJbREhghzL_seq2" "B6ByNegPMK_seq0" "B6ByNegPMK_seq1" "B6ByNegPMK_seq2")

export ALGORITHMS=("depth_reg" "vloc" "pose_fusion" "pose_fusion_opt" "groundtruth")
export TOPICS=("/depth_reg/odometry" "/vloc/odometry" "/pose_fusion/odometry" "/pose_fusion/path_opt" "/habitat_camera/odometry")
# export ALGORITHMS=("vloc" "pose_fusion" "pose_fusion_opt" "groundtruth")
# export TOPICS=("/vloc/odometry" "/pose_fusion/odometry" "/pose_fusion/path_opt" "/habitat_camera/odometry")

for index in ${!SCENES[*]}; do
  export SCENE=${SCENES[$index]}
  export SEQ_NAME=${SEQ_NAMES[$index]}
  for index_algo in ${!ALGORITHMS[*]}; do
    if [ "${ALGORITHMS[$index_algo]}" = "groundtruth" ]; then
      export EVAL_PATH=$DATA_PATH/../vloc_eval_data/groundtruth
      python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
      --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_$SEQ_NAME.bag \
      --out_pose_path $EVAL_PATH/traj/$SEQ_NAME.txt \
      --topic_odom ${TOPICS[$index_algo]}
      continue
    elif [ "${ALGORITHMS[$index_algo]}" = "depth_reg" ]; then
      export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
      python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
      --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_$SEQ_NAME.bag \
      --out_pose_path $EVAL_PATH/matterport_${ALGORITHMS[$index_algo]}/laptop/traj/"$SEQ_NAME"_local.txt \
      --topic_odom ${TOPICS[$index_algo]}
      continue
    elif [ "${ALGORITHMS[$index_algo]}" = "pose_fusion_opt" ]; then
      export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
      python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
      --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_$SEQ_NAME.bag \
      --out_pose_path $EVAL_PATH/matterport_${ALGORITHMS[$index_algo]}/laptop/traj/$SEQ_NAME.txt \
      --topic_path ${TOPICS[$index_algo]}
      continue
    else
      export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
      python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
      --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_$SEQ_NAME.bag \
      --out_pose_path $EVAL_PATH/matterport_${ALGORITHMS[$index_algo]}/laptop/traj/$SEQ_NAME.txt \
      --topic_odom ${TOPICS[$index_algo]}
    fi
  done
done