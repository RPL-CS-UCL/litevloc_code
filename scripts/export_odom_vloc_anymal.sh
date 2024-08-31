export PYTHON_PROJECT_PATH=/Titan/code/robohike_ws/src/pycpptools/
export DATA_PATH=/Rocket_ssd/dataset/data_topo_loc/ucl_campus

export SCENES=("ops_msg" "ops_msg" "ops_msg")
export SEQ_NAMES=("ops_msg_seq0" "ops_msg_seq1" "ops_msg_seq2")

export LOCAL_ODOMETRY=("robotodom" "airslam")
export ALGORITHMS=("robotodom" "airslam" "vloc" "pose_fusion" "pose_fusion_opt")
export TOPICS=("/state_estimator/odometry" "/AirSLAM/odometry" "/vloc/odometry" "/pose_fusion/odometry" "/pose_fusion/path_opt")

for index in ${!SCENES[*]}; do
  export SCENE=${SCENES[$index]}
  export SEQ_NAME=${SEQ_NAMES[$index]}
  for lo in ${!LOCAL_ODOMETRY[*]}; do
    # With robot_odometry as the local odometry
    if [ "${LOCAL_ODOMETRY[$lo]}" = "robotodom" ]; then
      for index_algo in ${!ALGORITHMS[*]}; do
        export ALGORITHM=${ALGORITHMS[$index_algo]}
        # if [ "$ALGORITHM" = "pose_fusion_opt" ]; then
        #   export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
        #   python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
        #   --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_robotodom_$SEQ_NAME.bag \
        #   --out_pose_path $EVAL_PATH/anymal_robotodom_$ALGORITHM/laptop/traj/"$SEQ_NAME"_zed.txt \
        #   --topic_path ${TOPICS[$index_algo]}
        # else
        #   export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
        #   python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
        #   --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_robotodom_$SEQ_NAME.bag \
        #   --out_pose_path $EVAL_PATH/anymal_robotodom_$ALGORITHM/laptop/traj/"$SEQ_NAME"_zed.txt \
        #   --topic_odom ${TOPICS[$index_algo]}
        # fi
      done
    # With airslam as the local odometry
    elif [ "${LOCAL_ODOMETRY[$lo]}" = "airslam" ]; then
      for index_algo in ${!ALGORITHMS[*]}; do
        export ALGORITHM=${ALGORITHMS[$index_algo]}
        # Export odometry from ROSbag files
        if [ "$ALGORITHM" = "pose_fusion_opt" ]; then
          export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
          python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
          --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_airslam_$SEQ_NAME.bag \
          --out_pose_path $EVAL_PATH/anymal_airslam_$ALGORITHM/laptop/traj/"$SEQ_NAME"_zed.txt \
          --topic_path ${TOPICS[$index_algo]}
        else
          export EVAL_PATH=$DATA_PATH/../vloc_eval_data/algorithms
          python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_bag_save_odom.py \
          --in_bag_path $DATA_PATH/vloc_$SCENE/$SEQ_NAME/loc_result_airslam_$SEQ_NAME.bag \
          --out_pose_path $EVAL_PATH/anymal_airslam_$ALGORITHM/laptop/traj/"$SEQ_NAME"_zed.txt \
          --topic_odom ${TOPICS[$index_algo]}
        fi
        # Transform the odometry to the frame consistent with the GT for evaluation
        python $PYTHON_PROJECT_PATH/src/python/utils_ros/tools_transform_odom.py \
        --in_odom_path $EVAL_PATH/anymal_airslam_$ALGORITHM/laptop/traj/"$SEQ_NAME"_zed.txt \
        --out_odom_path $EVAL_PATH/anymal_airslam_$ALGORITHM/laptop/traj/"$SEQ_NAME".txt
      done
    fi
  done
done