#! /usr/bin/env python

"""
Usage: 
python ros_global_planner.py \
	--map_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--image_size 512 288 --device=cuda \
	--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 --save_descriptors \
	--num_preds_to_save 3 
"""

import os
import pathlib
import rospy

from loc_pipeline import LocPipeline
from utils.utils_pipeline import parse_arguments

from global_planner import GlobalPlanner

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.map_path, 'tmp/output_ros_global_planner'))

	# Initialize the global planner
	global_planner = GlobalPlanner(args)
	global_planner.read_trav_graph_from_files()

	# Initialize the localization pipeline
	global_planner.loc_pipeline = LocPipeline(args, out_dir)
	global_planner.loc_pipeline.init_vpr_model()
	global_planner.loc_pipeline.read_covis_graph_from_files()
	
	rospy.init_node('ros_global_planner', anonymous=True)
	global_planner.loc_pipeline.initalize_ros()
	global_planner.initalize_ros()
	global_planner.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	global_planner.main_freq = rospy.get_param('~main_freq', 1)
	global_planner.conv_dist = rospy.get_param('~conv_dist', 0.5)

	rospy.spin()
