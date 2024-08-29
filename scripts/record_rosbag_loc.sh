rosbag record -o /Rocket_ssd/dataset/data_topo_loc/loc_result \
  /habitat_camera/odometry \
  /graph /graph/poses \
  /vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs \
  /pose_fusion/odometry /pose_fusion/path /pose_fusion/path_opt \
  /depth_reg/odometry /depth_reg/path \
  /tf /tf_static
