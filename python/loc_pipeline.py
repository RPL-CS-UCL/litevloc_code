#!/usr/bin/env python

import os
import sys
import pathlib
import logging

import numpy as np
import torch

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf.transformations as tf

import pycpptools.src.python.utils_algorithm as pytool_alg
import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros

from matching.utils import to_numpy

from utils.utils_vpr_method import initialize_vpr_model, perform_knn_search
from utils.utils_vpr_method import save_visualization as save_vpr_visualization
from utils.utils_image_matching_method import initialize_img_matcher, compute_scale_factor
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image_matching_method import save_output as save_img_matcher_output
from utils.utils_pipeline import *

class LocPipeline:
	def __init__(self):
  	# Initialize arguments for algorithms
		self.args = parse_arguments()

		# Setup logging path
		out_dir = pathlib.Path(os.path.join(self.args.dataset_path, 'output_loc_pipeline'))
		out_dir.mkdir(exist_ok=True, parents=True)
		self.log_dir = setup_log_environment(out_dir.as_posix(), self.args)

		# Initialize system variables
		self.has_global_position = False
		self.has_local_position = False

		self.loc_start_node = None
		self.loc_goal_node = None
		self.current_node = None

		# Read models for algorithms
		self.vpr_model = initialize_vpr_model(self.args.vpr_method, 
																				  self.args.vpr_backbone, 
																					self.args.vpr_descriptors_dimension,
																					self.args.device)
		self.img_matcher = initialize_img_matcher(self.args.img_matcher, 
																						  self.args.device, 
																						  self.args.n_kpts)

		# Setup ROS publishers
		self.pub_graph = rospy.Publisher('/image_graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/image_graph/poses', PoseArray, queue_size=10)

		self.pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/path', Path, queue_size=10)
		self.br = tf2_ros.TransformBroadcaster()

		self.path_msg = Path()

	def load_data(self):
		# Read map and observations
		data_path = os.path.join(self.args.dataset_path, 'map')
		self.image_graph = load_map(data_path, self.args.image_size, self.args.depth_scale, 
																normalized=False, num_sample=self.args.sample_map)
		logging.info(f"Loaded {self.image_graph.get_num_node()} map nodes from {data_path}.")

		data_path = os.path.join(self.args.dataset_path, 'obs')
		self.image_obs = load_map(data_path, self.args.image_size, self.args.depth_scale, 
															normalized=False, num_sample=self.args.sample_obs)
		logging.info(f"Loaded {self.image_graph.get_num_node()} map nodes from {data_path}.")		

	def extract_vpr_descriptor(self, node):
		with torch.inference_mode():
			logging.info("Extracting descriptors for evaluation/testing")
			descriptor = self.vpr_model(node.rgb_image.unsqueeze(0).to(self.args.device))
			node.set_descriptor(descriptor)

	def perform_image_matching(self, map_node, obs_node):
		"""
		Perform image matching using map node and current node.
		"""
		map_id, obs_id = map_node.id, obs_node.id
		try:
			out_str = f"Paths: map_id ({map_id}), obs_id ({obs_id}). "
			matcher_result = self.img_matcher(map_node.rgb_image, obs_node.rgb_image)
			num_inliers, H, mkpts0, mkpts1 = (
				matcher_result["num_inliers"],
				matcher_result["H"],
				matcher_result["inliers0"],
				matcher_result["inliers1"],
			)
			assert num_inliers > 100
			
			"""Save matching results"""
			out_str += f"Found {num_inliers} inliers after RANSAC. "
			viz_path = save_img_matcher_visualization(map_node.rgb_image, obs_node.rgb_image, 
																								mkpts0, mkpts1, self.log_dir, obs_id, n_viz=100)
			out_str += f"Viz saved in {viz_path}. "
			dict_path = save_img_matcher_output(matcher_result, None, None, 
													 								self.args.img_matcher, self.args.n_kpts, 
																					self.args.image_size, self.log_dir, obs_id)
			out_str += f"Output saved in {dict_path}"       
		except Exception as e:
			print(f"Error in Matching: {e} due to no overlapping regions or insufficient matching.")
			return None
		print(out_str)
		scene = self.img_matcher.scene

		##### Scale predict depth images
		depth_img_meas = np.squeeze(np.transpose(to_numpy(obs_node.depth_image), (1, 2, 0)), axis=2) # 1xHXW -> HxWx1
		depth_img_est = to_numpy(scene.get_depthmaps())[1]
		mask = (depth_img_meas < self.args.min_depth_pro) | (depth_img_meas > self.args.max_depth_pro)
		depth_img_meas[mask] = 0.0
		depth_img_est[mask] = 0.0
		meas_scale = compute_scale_factor(depth_img_meas, depth_img_est)

		##### Retrieve estimated transformation matrix and correct the scale
		im_poses = to_numpy(scene.get_im_poses())
		if abs(np.sum(np.diag(im_poses[1])) - 4.0) < 1e-5:
			est_T_map_obs = np.linalg.inv(im_poses[0])
		else:
			est_T_map_obs = im_poses[1]
		est_T_map_obs[:3, 3] *= meas_scale
		
		matcher_result["meas_scale"] = meas_scale
		matcher_result["est_T_map_obs"] = est_T_map_obs
		return matcher_result

	def publish_message(self):
		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id = "map"

		# Publish image graph
		pytool_ros.ros_vis.publish_graph(self.image_graph, header, self.pub_graph, self.pub_graph_poses)

		# Publish odometry, path and tf messages
		if self.current_node is not None:
			child_frame_id = "camera"

			odom_msg = pytool_ros.ros_msg.convert_vec_to_rosodom(
				self.current_node.trans_w_node, 
				self.current_node.quat_w_node, 
				header, child_frame_id
			)
			self.pub_odom.publish(odom_msg)
			
			pose_msg = pytool_ros.ros_msg.convert_vec_to_rospose(
				self.current_node.trans_w_node, 
				self.current_node.quat_w_node, 
				header
			)
			self.path_msg.header = header
			self.path_msg.poses.append(pose_msg)
			self.pub_path.publish(self.path_msg)

			tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(
				self.current_node.trans_w_node, 
				self.current_node.quat_w_node, 
				header, child_frame_id
			)
			self.br.sendTransform(tf_msg)		

	def run(self):
		##### Create edges between nodes in the graph
		for map_id, _ in self.image_graph.nodes.items():
			if map_id == 0: continue
			map_node_prev = self.image_graph.get_node(map_id - self.args.sample_map)
			map_node_next = self.image_graph.get_node(map_id)
			if (map_node_prev is not None) and (map_node_next is not None):
				weight, _ = pytool_math.tools_eigen.compute_relative_dis(
					map_node_prev.trans_w_node, map_node_prev.quat_w_node, 
					map_node_next.trans_w_node, map_node_next.quat_w_node,
					mode='xyzw')
				self.image_graph.add_edge(map_node_prev, map_node_next, weight)

		##### Extract VPR descriptors of map nodes
		db_descriptors_id = self.image_graph.get_all_id()
		db_descriptors = np.empty((self.image_graph.get_num_node(), self.args.vpr_descriptors_dimension), dtype="float32")
		for indices, (_, map_node) in enumerate(self.image_graph.nodes.items()):
			self.extract_vpr_descriptor(map_node)
			db_descriptors[indices] = map_node.get_descriptor().cpu().numpy()
		print(f"IDs: {db_descriptors_id} extracted {len(db_descriptors)} VPR descriptors.")
		if self.args.save_descriptors:
			save_descriptors(self.log_dir, db_descriptors, desc_name="database_descriptors")

		# DEBUG(gogojjh)
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.publish_message()
			rate.sleep()

		##### Main loop: receive camera images, perform global and local localization, and publish messages
		if self.loc_goal_node is None:
			self.loc_goal_node = self.image_graph.get_node(10)

		rate = rospy.Rate(10) # 10 Hz
		for obs_id, obs_node in self.image_obs.nodes.items():
			if obs_id > 10:
				break

			##### Perform global localization
			if not self.has_global_position:
				query_descriptor = np.empty((1, self.args.vpr_descriptors_dimension), dtype="float32")
				self.extract_vpr_descriptor(obs_node)
				query_descriptor[0] = obs_node.get_descriptor().cpu().numpy()
				vpr_result = perform_knn_search(db_descriptors, query_descriptor, 
																				self.args.vpr_descriptors_dimension, 
																				self.args.recall_values)[0]
				##### Save VPR results
				if self.args.num_preds_to_save != 0:
					list_of_images_paths = [obs_node.rgb_img_path]
					for i in range(len(vpr_result[:self.args.num_preds_to_save])):
						map_node = self.image_graph.get_node(db_descriptors_id[vpr_result[i]])
						list_of_images_paths.append(map_node.rgb_img_path)
					preds_correct = [None] * len(list_of_images_paths)
					save_vpr_visualization(self.log_dir, obs_id, list_of_images_paths, preds_correct)

				##### Use VPR results to update global position and create global path
				if len(vpr_result) > 0:
					matched_map_node = self.image_graph.get_node(db_descriptors_id[vpr_result[0]])
					if matched_map_node is not None:
						tra_distance, tra_path = \
							pytool_alg.sp.dijk_shortest_path(self.image_graph, matched_map_node, self.loc_goal_node)
						if tra_distance == float('inf'):
							print('No path found between start and goal nodes.')
							continue

						##### Existing path from start to goal node
						self.has_global_position = True
						self.loc_start_node = matched_map_node
						self.current_node = self.loc_start_node
						for i in range(len(tra_path) - 1):
							node = tra_path[i]
							node_next = tra_path[i + 1]
							node.add_next_node(node_next)
						out_str  = f"Found matching between {obs_id} and {matched_map_node.id}\n"
						out_str += f"Travel distance of the shortest path: {tra_distance:.3f}m\n"
						out_str += f"Start traveling from {self.loc_start_node.id} -> {self.loc_goal_node.id}\n"
						out_str += f"Shortest path: " + " -> ".join([str(node.id) for node in tra_path])
						print(out_str)
					else:
						# No matching map node found
						continue
			
			##### Perform local localization if global localization is available
			subgoal_node = self.current_node.get_next_node()
			matcher_result = self.perform_image_matching(subgoal_node, obs_node)
			if matcher_result is not None:
				# Get the groundtruth pose
				T_w_map = pytool_math.tools_eigen.convert_vec_to_matrix(subgoal_node.trans_w_node, subgoal_node.quat_w_node, 'xyzw')
				T_w_obs = pytool_math.tools_eigen.convert_vec_to_matrix(obs_node.trans_w_node, obs_node.quat_w_node, 'xyzw')
				T_map_obs = np.linalg.inv(T_w_map) @ T_w_obs
				# Get the estimated pose
				trans_map_obs, quat_map_obs = pytool_math.tools_eigen.convert_matrix_to_vec(matcher_result["est_T_map_obs"], 'xyzw')
				self.current_node.set_pose(trans_map_obs, quat_map_obs)
				print('Groundtruth Poses:\n', T_map_obs)
				print(f'Estimated Poses with Meas scale {matcher_result["meas_scale"]}:\n', matcher_result["est_T_map_obs"])
				if not self.args.no_viz:
					self.img_matcher.scene.show(cam_size=0.05)
			else:
				continue

			##### Publish path
			self.publish_message()

		rate.sleep() # 10Hz

if __name__ == '__main__':
	rospy.init_node('loc_pipeline_node', anonymous=True)

	loc_pipeline = LocPipeline()
	loc_pipeline.load_data()
	loc_pipeline.run()