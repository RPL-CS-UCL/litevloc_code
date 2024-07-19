#!/usr/bin/env python

"""
Usage: python global_planner.py --start_node_id 0 \
--dataset_path /Titan/dataset/data_topo_loc/anymal_ops_mos --image_size 288 512 --device=cuda \
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
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros
import argparse
from datetime import datetime

import pycpptools.src.python.utils_algorithm as pytool_alg
import pycpptools.src.python.utils_ros as pytool_ros

from utils.utils_vpr_method import initialize_vpr_model, perform_knn_search, save_visualization as save_vpr_visualization
from utils.utils_image import load_rgb_image
from point_graph import PointGraphLoader as GraphLoader

class GlobalPlanner:
	def __init__(self, args, log_dir):
		self.args = args
		self.log_dir = log_dir

		# Initialize variables
		self.plan_goal_node = None
		self.plan_start_node = None
		self.planner_path = []

		# VPR models
		self.vpr_model = initialize_vpr_model(self.args.vpr_method, 
																					self.args.vpr_backbone, 
																					self.args.vpr_descriptors_dimension,
																					self.args.device)

		# ROS publisher
		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		self.pub_shortest_path = rospy.Publisher('/graph/shortest_path', MarkerArray, queue_size=10)
		self.br = tf2_ros.TransformBroadcaster()

	def read_map_from_file(self):
		data_path = os.path.join(self.args.dataset_path, 'map')
		self.point_graph = GraphLoader.load_data(data_path)
		logging.info(f"Loaded {self.point_graph} from {data_path}")

	def perform_vpr(self, db_descs, query_desc):
		query_desc_arr = np.empty((1, self.args.vpr_descriptors_dimension), dtype="float32")
		query_desc_arr[0] = query_desc
		dis, pred = perform_knn_search(
			db_descs,
			query_desc_arr,
			self.args.vpr_descriptors_dimension,
			self.args.recall_values
		)
		return dis, pred

	def publish_message(self):
		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id = "map"

		tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, 'map_graph')
		self.br.sendTransform(tf_msg)
		header.frame_id = 'map_graph'
		pytool_ros.ros_vis.publish_graph(self.point_graph, header, self.pub_graph, self.pub_graph_poses)
		if self.planner_path:
			pytool_ros.ros_vis.publish_shortest_path(self.planner_path, header, self.pub_shortest_path)

	def run(self):
		rospy.init_node('loc_pipeline_node', anonymous=True)

		# Extract VPR descriptors for all nodes in the map
		db_descriptors_id = self.point_graph.get_all_id()
		db_descriptors = np.array([map_node.get_descriptor() for _, map_node in self.point_graph.nodes.items()], dtype="float32")
		print(f"IDs: {db_descriptors_id} extracted {db_descriptors.shape} VPR descriptors.")

		"""Extract VPR descriptors for the goal nodes"""
		vpr_start_time = time.time()
		goal_img_path = os.path.join(self.args.dataset_path, 'map', 'goal.png')
		goal_img = load_rgb_image(goal_img_path, self.args.image_size, normalized=False)
		with torch.no_grad():
			desc = self.vpr_model(goal_img.unsqueeze(0).to(self.args.device)).cpu().numpy()
		vpr_dis, vpr_pred = self.perform_vpr(db_descriptors, desc)
		vpr_dis, vpr_pred = vpr_dis[0, :], vpr_pred[0, :]		
		if len(vpr_pred) == 0:
			print('No goal node found, cannot determine the global position of the goal.')
			return
		
		out_str  = 'Top-K VPR results:\n'
		out_str += 'Matched ID: ' + ', '.join([f"{id}" for id in vpr_pred]) + '\n'
		out_str += 'Distance: ' + ', '.join([f"{d:.3f}" for d in vpr_dis]) + '\n'
		out_str += f'Time taken for VPR: {time.time() - vpr_start_time:.3f}s'
		print(out_str)

		# Save VPR visualization for the top-k predictions
		if self.args.num_preds_to_save != 0:
			list_of_images_paths = [goal_img_path]
			for i in range(len(vpr_pred[:self.args.num_preds_to_save])):
				map_node = self.point_graph.get_node(db_descriptors_id[vpr_pred[i]])
				list_of_images_paths.append(map_node.rgb_img_path)
			preds_correct = [None] * len(list_of_images_paths)
			save_vpr_visualization(self.log_dir, 0, list_of_images_paths, preds_correct)

		"""Shortest path planning"""
		self.plan_start_node = self.point_graph.get_node(self.args.start_node_id)
		if self.plan_start_node is None:
			print('No start node found.')
			return

		spath_start_time = time.time()
		goal_node = self.point_graph.get_node(db_descriptors_id[vpr_pred[0]])
		if goal_node is not None:
			tra_distance, tra_path = pytool_alg.sp.dijk_shortest_path(self.point_graph, self.plan_start_node, goal_node)
			if tra_distance == float('inf'):
				print('No path found between start and goal nodes.')
				return

		self.plan_goal_node = goal_node
		self.planner_path = tra_path
		for i in range(len(tra_path) - 1):
			node = tra_path[i]
			node_next = tra_path[i + 1]
			node.add_next_node(node_next)

		out_str =  f"Travel distance of the shortest path: {tra_distance:.3f}m\n"
		out_str += f"Start traveling from {self.plan_start_node.id} -> {self.plan_goal_node.id}\n"
		out_str += f"Shortest path: " + " -> ".join([str(node.id) for node in tra_path])
		out_str += f"\nTime taken for shortest path planning: {time.time() - spath_start_time:.3f}s"
		print(out_str)

		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.publish_message()
			rate.sleep()

def parse_arguments():
	parser = argparse.ArgumentParser(description="Global Planner", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
											help="Resizing shape for images (HxW). If a single int is passed, set the"
											"smallest edge of all images to this value, while keeping aspect ratio")
	parser.add_argument("--start_node_id", type=int, default=0, help="ID of the start node")

	"""
	Parameters for VPR methods
	"""
	parser.add_argument("--positive_dist_threshold", type=int, default=25,
											help="distance (in meters) for a prediction to be considered a positive")
	parser.add_argument("--vpr_method", type=str, default="cosplace",
											choices=["netvlad", "apgem", "sfrs", "cosplace", "convap", "mixvpr", "eigenplaces", 
																"eigenplaces-indoor", "anyloc", "salad", "salad-indoor", "cricavpr"],
											help="_")
	parser.add_argument("--vpr_backbone", type=str, default=None,
											choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
											help="_")
	parser.add_argument("--vpr_descriptors_dimension", type=int, default=None,
											help="_")
	
	parser.add_argument("--num_workers", type=int, default=4,
											help="_")
	parser.add_argument("--batch_size", type=int, default=4,
											help="set to 1 if database images may have different resolution")
	parser.add_argument("--log_dir", type=str, default="default",
											help="experiment name, output logs will be saved under logs/log_dir")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
											help="_")
	parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
											help="values for recall (e.g. recall@1, recall@5)")
	parser.add_argument("--no_labels", action="store_true",
											help="set to true if you have no labels and just want to "
											"do standard image retrieval given two folders of queries and DB")
	parser.add_argument("--num_preds_to_save", type=int, default=0,
											help="set != 0 if you want to save predictions for each query")
	parser.add_argument("--save_only_wrong_preds", action="store_true",
											help="set to true if you want to save predictions only for "
											"wrongly predicted queries")
	parser.add_argument("--save_descriptors", action="store_true",
											help="set to True if you want to save the descriptors extracted by the model")	
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_global_planner'))
	out_dir.mkdir(exist_ok=True, parents=True)
	start_time = datetime.now()
	tmp_dir = os.path.join(out_dir, f'outputs_{args.vpr_method}')
	log_dir = os.path.join(tmp_dir, f'{args.vpr_backbone}_' + start_time.strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(os.path.join(log_dir, 'preds'))
	os.system(f"rm {os.path.join(tmp_dir, 'latest')}")
	os.system(f"ln -s {log_dir} {os.path.join(tmp_dir, 'latest')}")

	global_planner = GlobalPlanner(args, log_dir)
	global_planner.read_map_from_file()
	global_planner.run()
