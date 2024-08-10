rosbag record -o /Rocket_ssd/dataset/data_topo_loc/vloc \
  /habitat_camera/color/camera_info /habitat_camera/depth/camera_info \
  /habitat_camera/color/image /habitat_camera/depth/image \
  /habitat_camera/odometry \
  /graph /graph/poses \
  /vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs \
  /tf /tf_static
