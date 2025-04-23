#! /usr/bin/env python

"""
Usage: 
python ros_global_planner.py \
--dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
--image_size 288 512 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors \
--num_preds_to_save 3 
"""

import os
import pathlib
import numpy as np
import rospy

from loc_pipeline import LocPipeline
from utils.utils_pipeline import parse_arguments, setup_log_environment

from global_planner import GlobalPlanner

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.map_path, 'output_ros_global_planner'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the global planner
	global_planner = GlobalPlanner(args)
	global_planner.read_map_from_file()

	# Initialize the localization pipeline
	global_planner.loc_pipeline = LocPipeline(args, log_dir)
	global_planner.loc_pipeline.init_vpr_model()
	global_planner.loc_pipeline.read_map_from_file()
	
	rospy.init_node('ros_global_planner', anonymous=True)
	global_planner.loc_pipeline.initalize_ros()
	global_planner.initalize_ros()
	global_planner.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	global_planner.main_freq = rospy.get_param('~main_freq', 1)
	global_planner.conv_dist = rospy.get_param('~conv_dist', 0.5)

	rospy.spin()
