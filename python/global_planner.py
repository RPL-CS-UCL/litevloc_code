"""
Usage: 
python global_planner.py \
--dataset_path /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_17DRP5sb8fy/out_map \
--image_size 288 512 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors --num_preds_to_save 3 
"""

import os
import pathlib
import logging
import numpy as np
import torch
import time
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from point_graph import PointGraphLoader as GraphLoader
from image_node import ImageNode
from loc_pipeline import LocPipeline
from utils.utils_image import load_rgb_image
from utils.utils_pipeline import parse_arguments, setup_log_environment

import pycpptools.src.python.utils_algorithm as pytool_alg
import pycpptools.src.python.utils_ros as pytool_ros

class GlobalPlanner:
	def __init__(self, args, log_dir):
		self.args = args
		self.log_dir = log_dir

		# Initialize variables
		self.plan_start_node = None
		self.plan_goal_node = None
		self.planner_path = []

	def setup_ros_objects(self):
		# ROS publisher
		self.pub_shortest_path = rospy.Publisher('/graph/shortest_path', MarkerArray, queue_size=10)
		# self.pub_waypoint = rospy.Publisher('/graph/waypoint', MarkerArray, queue_size=10)

	def read_map_from_file(self):
		data_path = self.args.dataset_path
		self.point_graph = GraphLoader.load_data(data_path)
		logging.info(f"Loaded {self.point_graph} from {data_path}")

	def publish_message(self):
		if self.planner_path:
			header = Header(stamp=rospy.Time.now(), frame_id='map_graph')
			pytool_ros.ros_vis.publish_shortest_path(self.planner_path, header, self.pub_shortest_path)

			subgoal_node = self.planner_path[1]
			# pytool_ros.ros_vis.publish_waypoint(subgoal_node, header, self.pub_shortest_path)

def perform_planning(loc, gp, args):
	start_node_id = 0
	gp.plan_start_node = gp.point_graph.get_node(start_node_id)
	if gp.plan_start_node is None:
		print(f"Start node {start_node_id} not found in the graph.")
		return

	img_size = args.image_size
	goal_img_path = os.path.join(args.dataset_path, 'goal_image_0.jpg')
	goal_img = load_rgb_image(goal_img_path, img_size)
	with torch.no_grad():
		desc = loc.vpr_model(goal_img.unsqueeze(0).to(args.device)).cpu().numpy()
	obs_node = ImageNode(0, goal_img, None, desc, 
						0, np.zeros(3), np.array([0, 0, 0, 1]), 
						None, img_size, 
						goal_img_path, None)
	loc.curr_obs_node = obs_node

	spath_start_time = time.time()
	"""Perform global localization via. visual place recognition"""
	result = loc.perform_global_loc(save=True)
	loc.curr_obs_node = None # not influence the ros publish
	if not result['succ']:
		print('No goal node found via Global Localization.')
		return
	matched_map_id = result['map_id']
	gp.plan_goal_node = gp.point_graph.get_node(matched_map_id)
		
	"""Shortest path planning"""
	tra_distance, tra_path = pytool_alg.sp.dijk_shortest_path(gp.point_graph, gp.plan_start_node, gp.plan_goal_node)
	if tra_distance == float('inf'):
		print('No path found between start and goal nodes.')
		return
	gp.planner_path = tra_path
	for i in range(len(tra_path) - 1):
		node = tra_path[i]
		node_next = tra_path[i + 1]
		node.add_next_node(node_next)

	out_str =  f"Travel distance of the shortest path: {tra_distance:.3f}m\n"
	out_str += f"Start traveling from {gp.plan_start_node.id} -> {gp.plan_goal_node.id}\n"
	out_str += f"Shortest path: " + " -> ".join([str(node.id) for node in tra_path])
	out_str += f"\nTime taken for shortest path planning: {time.time() - spath_start_time:.3f}s"
	print(out_str)

	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		loc.publish_message()
		gp.publish_message()
		rate.sleep()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_global_planner'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, log_dir)
	loc_pipeline.init_vpr_model()
	loc_pipeline.read_map_from_file()

	# Initialize the global planner
	global_planner = GlobalPlanner(args, log_dir)
	global_planner.read_map_from_file()
	
	rospy.init_node('global_planner', anonymous=True)
	loc_pipeline.setup_ros_objects()
	global_planner.setup_ros_objects()
	
	perform_planning(loc_pipeline, global_planner, args)
