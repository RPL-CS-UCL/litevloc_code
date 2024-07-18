#!/usr/bin/env python

import os
import sys
import pathlib
import logging

import numpy as np
import torch
import copy

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

		self.curr_subgoal_node = None
		self.curr_obs_node = None

		# Read models for algorithms
		self.vpr_model = initialize_vpr_model(self.args.vpr_method, 
																				  self.args.vpr_backbone, 
																					self.args.vpr_descriptors_dimension,
																					self.args.device)
		self.img_matcher = initialize_img_matcher(self.args.img_matcher, 
																						  self.args.device, 
																						  self.args.n_kpts)

		# Setup ROS publishers and message
		self.pub_graph = rospy.Publisher('/image_graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/image_graph/poses', PoseArray, queue_size=10)

		self.pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/path', Path, queue_size=10)
		self.pub_path_gt = rospy.Publisher('/path_gt', Path, queue_size=10)

		self.pub_img_goal_obs = rospy.Publisher('/image_subgoal_obs', Image, queue_size=10)

		self.br = tf2_ros.TransformBroadcaster()

		self.path_msg = Path()
		self.path_gt_msg = Path()

	def load_data(self):
		# Read map and observations
		data_path = os.path.join(self.args.dataset_path, 'map')
		self.image_graph = ImageGraphLoader.load_data(
			data_path, self.args.image_size, self.args.depth_scale,
			normalized=False, num_sample=self.args.sample_map
		)
		logging.info(f"Loaded {self.image_graph.get_num_node()} map nodes from {data_path}.")

		data_path = os.path.join(self.args.dataset_path, 'obs')
		self.image_obs = ImageGraphLoader.load_data(
			data_path, self.args.image_size, self.args.depth_scale, 
			normalized=False, num_sample=self.args.sample_obs, num_load=200
		)
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
			# out_str += f"Viz saved in {viz_path}. "
			dict_path = save_img_matcher_output(matcher_result, None, None, 
													 								self.args.img_matcher, self.args.n_kpts, 
																					self.args.image_size, self.log_dir, obs_id)
			# out_str += f"Output saved in {dict_path}"       
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
			est_T_subgoal_obs = np.linalg.inv(im_poses[0])
		else:
			est_T_subgoal_obs = im_poses[1]
		est_T_subgoal_obs[:3, 3] *= meas_scale
		
		matcher_result["meas_scale"] = meas_scale
		matcher_result["est_T_subgoal_obs"] = est_T_subgoal_obs
		return matcher_result

	def publish_message(self):
		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id = "map"

		# Publish image graph
		pytool_ros.ros_vis.publish_graph(self.image_graph, header, self.pub_graph, self.pub_graph_poses)

		# Publish odometry, path and tf messages
		if self.curr_obs_node is not None:
			child_frame_id = "camera"

			odom_msg = pytool_ros.ros_msg.convert_vec_to_rosodom(
				self.curr_obs_node.trans, self.curr_obs_node.quat, header, child_frame_id)
			self.pub_odom.publish(odom_msg)
			
			pose_msg = pytool_ros.ros_msg.convert_vec_to_rospose(
				self.curr_obs_node.trans, self.curr_obs_node.quat, header)
			self.path_msg.header = header
			self.path_msg.poses.append(pose_msg)
			self.pub_path.publish(self.path_msg)

			tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(
				self.curr_obs_node.trans, self.curr_obs_node.quat, header, child_frame_id)
			self.br.sendTransform(tf_msg)		

			if self.curr_obs_node.has_pose_gt:
				pose_msg = pytool_ros.ros_msg.convert_vec_to_rospose(
					self.curr_obs_node.trans_gt, self.curr_obs_node.quat_gt, header)			
				self.path_gt_msg.header = header
				self.path_gt_msg.poses.append(pose_msg)
				self.pub_path_gt.publish(self.path_gt_msg)

			if self.curr_subgoal_node is not None:
				rgb_img_obs = np.transpose(to_numpy(self.curr_obs_node.rgb_image), (1, 2, 0))
				rgb_img_subgoal = np.transpose(to_numpy(self.curr_subgoal_node.rgb_image), (1, 2, 0))
				rgb_img_merge = np.hstack((rgb_img_subgoal, rgb_img_obs))
				img_msg = pytool_ros.ros_msg.convert_cvimg_to_rosimg(rgb_img_merge, "bgr8", header, compressed=False)
				self.pub_img_goal_obs.publish(img_msg)

	def run(self):
		##### Create edges between nodes in the graph
		for map_id, _ in self.image_graph.nodes.items():
			if map_id == 0: continue
			map_node_prev = self.image_graph.get_node(map_id - self.args.sample_map)
			map_node_next = self.image_graph.get_node(map_id)
			if (map_node_prev is not None) and (map_node_next is not None):
				weight, _ = pytool_math.tools_eigen.compute_relative_dis(
					map_node_prev.trans_gt, map_node_prev.quat_gt, 
					map_node_next.trans_gt, map_node_next.quat_gt,
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

		##### Main loop: receive camera images, perform global and local localization, and publish messages
		if self.loc_goal_node is None:
			self.loc_goal_node = self.image_graph.get_node(10)

		rate = rospy.Rate(10) # 10 Hz
		for obs_id, obs_node in self.image_obs.nodes.items():
			self.curr_obs_node = obs_node

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
						for i in range(len(tra_path) - 1):
							node = tra_path[i]
							node_next = tra_path[i + 1]
							node.add_next_node(node_next)
						self.loc_start_node = matched_map_node
						self.curr_subgoal_node = matched_map_node.get_next_node()

						out_str  = f"Found matching between {obs_id} and {matched_map_node.id}\n"
						out_str += f"Travel d-istance of the shortest path: {tra_distance:.3f}m\n"
						out_str += f"Start traveling from {self.loc_start_node.id} -> {self.loc_goal_node.id}\n"
						out_str += f"Shortest path: " + " -> ".join([str(node.id) for node in tra_path])
						print(out_str)
					else:
						# No matching map node found
						continue
			
			##### Perform local localization if global localization is available
			matcher_result = self.perform_image_matching(self.curr_subgoal_node, self.curr_obs_node)
			if matcher_result is not None:
				# Get the groundtruth pose
				print(f'Groundtruth Poses: {self.curr_obs_node.trans_gt.T}')

				# Get the estimated pose
				meas_scale, est_T_subgoal_obs = matcher_result["meas_scale"], matcher_result["est_T_subgoal_obs"]
				T_w_subgoal = pytool_math.tools_eigen.convert_vec_to_matrix(self.curr_subgoal_node.trans_gt, self.curr_subgoal_node.quat_gt, 'xyzw')
				T_w_obs = T_w_subgoal @ est_T_subgoal_obs
				trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(T_w_obs, 'xyzw')
				self.curr_obs_node.set_pose(trans, quat)
				print(f'Estimated Poses with Meas scale {meas_scale:.3f}: {trans.T}\n')

				# Compute distance between the subgoal and current observation
				if self.curr_subgoal_node.get_next_node() is not None:
					dis_trans, dis_angle = pytool_math.tools_eigen.compute_relative_dis_TF(est_T_subgoal_obs, np.eye(4))
					if dis_trans < 0.2 and dis_angle < 20.0:
						print(f"Switch subgoal from {self.curr_subgoal_node.id} to {self.curr_subgoal_node.get_next_node().id}")
						self.curr_subgoal_node = self.curr_subgoal_node.get_next_node()

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