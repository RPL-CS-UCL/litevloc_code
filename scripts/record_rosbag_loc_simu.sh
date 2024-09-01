rosbag record -o /Rocket_ssd/dataset/data_topo_loc/navigation_result \
  /habitat_camera/color/image /habitat_camera/depth/image /habitat_camera/odometry \
  /goal_image /graph /graph/poses /graph/shortest_path \
  /vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs \
  /pose_fusion/odometry /pose_fusion/path /pose_fusion/path_opt \
  /depth_reg/odometry /depth_reg/path \
  /vloc/way_point /path /iplanner_path_fear /path_fear /iplanner_image \
  /overall_map /trajectory /cmd_vel /joy /way_point /navigation_boundary /state_estimation \
  /Odometry /Odometry_global \
  /AirSLAM/odometry \
  /tf /tf_static

