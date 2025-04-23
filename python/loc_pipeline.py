#! /usr/bin/env python

"""
Usage: 
python python/loc_pipeline.py \
	--map_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--query_data_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--image_size 512 288 --device=cuda \
	--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 \
	--save_descriptors --num_preds_to_save 3 \
	--img_matcher master --save_img_matcher \
	--pose_solver pnp --config_pose_solver python/config/dataset/matterport3d.yaml \
	--global_pos_threshold 10.0 --min_inliers_threshold 300 --viz

Usage: 
rosbag record -O /Titan/dataset/data_litevloc/anymal_lab_upstair_20240722_0/vloc.bag \
	/vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs
"""

import os
import sys
import pathlib
import numpy as np
# import torch
import time
import cv2
import logging
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros
import matplotlib

from utils.utils_pipeline import *
from utils.utils_geom import read_intrinsics, read_poses, read_descriptors, correct_intrinsic_scale
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, compute_pose_error, convert_pose_inv
from utils.utils_vpr_method import initialize_vpr_model, perform_knn_search, compute_euclidean_dis
from utils.utils_vpr_method import save_visualization as save_vpr_visualization
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image import load_rgb_image, load_depth_image, to_numpy, save_rgb_image
from utils.utils_ros import ros_msg, ros_vis
from utils.pose_solver import get_solver
from benchmark_rpe.rpe_default import cfg
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

class LocPipeline:
	def __init__(self, args, out_dir):
		self.args = args

		out_dir.mkdir(exist_ok=True, parents=True)
		log_dir = setup_log_environment(out_dir, args)
		self.log_dir = log_dir
	
		self.has_global_pos = False
		self.has_local_pos = False
		self.frame_id_map = 'map'
		self.depth_range = (0.1, 15.0)

	def init_vpr_model(self):
		self.vpr_model = initialize_vpr_model(self.args.vpr_method, self.args.vpr_backbone, self.args.vpr_descriptors_dimension, self.args.device)
		logging.info(f"VPR model: {self.args.vpr_method}")

	def init_img_matcher(self):
		self.img_matcher = initialize_img_matcher(self.args.img_matcher, self.args.device, self.args.n_kpts)
		if self.args.img_matcher == "master": 
			self.img_matcher.min_conf_thr = self.args.min_master_conf_thre
		logging.info(f"Image matcher: {self.args.img_matcher}")
		
	def init_pose_solver(self):
		cfg.merge_from_file(self.args.config_pose_solver)
		self.pose_solver = get_solver(self.args.pose_solver, cfg)
		logging.info(f"Pose solver: {self.args.pose_solver}")

	def initalize_ros(self):
		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		
		self.pub_odom = rospy.Publisher('/vloc/odometry', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/vloc/path', Path, queue_size=10)
		self.pub_path_gt = rospy.Publisher('/vloc/path_gt', Path, queue_size=10)
		self.pub_map_obs = rospy.Publisher('/vloc/image_map_obs', Image, queue_size=10)

		self.br = tf2_ros.TransformBroadcaster()
		self.path_msg = Path()
		self.path_gt_msg = Path()

	def read_covis_graph_from_files(self):
		map_root = pathlib.Path(self.args.map_path)
		self.image_graph = GraphLoader.load_data(
			map_root=map_root,
			resize=self.args.image_size,
			depth_scale=self.args.depth_scale,
			load_rgb=True, 
			load_depth=True, 
			normalized=False,
			edge_type='covis'
		)
		logging.info(str(self.image_graph))

		# Extract VPR descriptors for all nodes in the map
		self.DB_DESCRIPTORS = np.array([node.get_descriptor() for _, node in self.image_graph.nodes.items()], dtype="float32")
		rospy.logdebug(f"Extracted {self.DB_DESCRIPTORS.shape} VPR descriptors from the map.")
		self.DB_POSES = np.empty((self.image_graph.get_num_node(), 7), dtype="float32")
		for indices, (_, node) in enumerate(self.image_graph.nodes.items()):
			self.DB_POSES[indices, :3] = node.trans
			self.DB_POSES[indices, 3:] = node.quat

	def perform_vpr(self, db_descs: np.array, query_desc: np.array):
		dis, pred = perform_knn_search(
			db_descs, query_desc,
			self.args.vpr_descriptors_dimension, 
			self.args.recall_values
		)
		return dis, pred

	def perform_image_matching(self, matcher, map_node, obs_node):
		try:
			matcher_result = matcher(map_node.rgb_image, obs_node.rgb_image)
			"""Save matching results"""
			if self.args.save_img_matcher:
				mkpts0, mkpts1 = \
					matcher_result["inlier_kpts0"], matcher_result["inlier_kpts1"],
				save_img_matcher_visualization(
					obs_node.rgb_image, map_node.rgb_image,
					mkpts0, mkpts1, 
					self.log_dir, 
					obs_node.id, 
					n_viz=100
				)	
			return matcher_result
		
		except Exception as e:
			logging.error(f"Error in image matching: {e}")
			null_kpts = np.zeros((0, 2), dtype=np.float32)
			return {"num_inliers_kpts": 0, "num_inliers": 0, "inliers0": null_kpts, "inliers1": null_kpts}

	def search_keyframe_from_graph(self, obs_node):
		"""
		This method searches for the visual-closest keyframe in the covisibility graph for a given observation node.

		Parameters:
		obs_node (Node): The observation node for which to find the closest keyframe. 

		Returns:
		Node: The closest keyframe node in the graph. 
			  If no keyframe is found within the global position threshold, it returns None.
		"""
		query_pose = obs_node.trans.reshape(1, 3)
		dis, pred = perform_knn_search(self.DB_POSES[:, :3], query_pose, 3, [1])
		if len(pred[0]) == 0 or dis[0][0] > self.args.global_pos_threshold: 
			return None
		
		closest_map_node = self.image_graph.get_node(pred[0][0])
		all_nei_nodes = [nei_node for nei_node, _ in closest_map_node.edges.values()] + [closest_map_node]
		list_dis = [compute_euclidean_dis(obs_node.get_descriptor(), node.get_descriptor()) for node in all_nei_nodes]
		node_min_dis = all_nei_nodes[np.argmin(list_dis)]
		out_str = 'Keyframe candidate: '
		out_str += ' '.join([f'{node.id}({dis:.2f})' for node, dis in zip(all_nei_nodes, list_dis)]) + f' Closest node: {node_min_dis.id}'
		rospy.loginfo(out_str)
		
		return node_min_dis

	def perform_global_loc(self, save_viz=False):
		query_desc = self.curr_obs_node.get_descriptor()
		_, vpr_pred = self.perform_vpr(self.DB_DESCRIPTORS, query_desc.reshape(1, -1))
		if save_viz:
			img_paths = [str(self.image_graph.map_root / self.curr_obs_node.rgb_img_name)]
			for i in range(len(vpr_pred[0, :self.args.num_preds_to_save])):
				map_node = self.image_graph.get_node(vpr_pred[0, i])
				img_paths.append(str(self.image_graph.map_root / map_node.rgb_img_name))
			preds_correct = [None] * len(img_paths)
			save_vpr_visualization(self.log_dir, 0, img_paths, preds_correct)
		
		return {'succ': True, 'map_id': vpr_pred[0, 0]}
	
	def perform_local_loc(self):
		result_fail = {'succ': False, 'T_w_obs': None, 'solver_inliers': 0}

		matching_start_time = time.time()
		ref_node = self.search_keyframe_from_graph(self.curr_obs_node)
		if ref_node is None: 
			return result_fail
		
		self.ref_map_node = ref_node
		match_result = self.perform_image_matching(self.img_matcher, self.ref_map_node, self.curr_obs_node)
		w_ratio = self.ref_map_node.raw_img_size[0] / self.ref_map_node.img_size[0] 
		h_ratio = self.ref_map_node.raw_img_size[1] / self.ref_map_node.img_size[1]
		mkpts0, mkpts1 = (match_result["inlier_kpts0"], match_result["inlier_kpts1"])
		mkpts0_raw = mkpts0 * [w_ratio, h_ratio]
		mkpts1_raw = mkpts1 * [w_ratio, h_ratio]
		
		num_inliers = match_result["num_inliers"]
		self.ref_map_node.set_matched_kpts(mkpts0, num_inliers)
		self.curr_obs_node.set_matched_kpts(mkpts1, num_inliers)
		rospy.loginfo(f'Number of matched inliers: {num_inliers}')
		rospy.loginfo(f"Image matching costs: {time.time() - matching_start_time: .3f}s")

		if num_inliers < self.args.min_kpts_inliers_thre:
			rospy.logwarn(f'[Fail] No sufficient matching kpts')
			return result_fail
		try:
			depth_img1 = to_numpy(self.curr_obs_node.depth_image.squeeze(0))
			R, t, num_solver_inliers = self.pose_solver.estimate_pose(
				mkpts1_raw, mkpts0_raw,
				self.curr_obs_node.raw_K, 
				self.ref_map_node.raw_K,
				depth_img1, None
			)
			if num_solver_inliers < self.args.min_solver_inliers_thre:
				rospy.logwarn(f'[Fail] No sufficient number {num_solver_inliers} solver inliers')
				return result_fail
			else:
				T_mapnode_obs = np.eye(4)
				T_mapnode_obs[:3, :3], T_mapnode_obs[:3, 3] = R, t.reshape(3)
				T_w_mapnode = convert_vec_to_matrix(self.ref_map_node.trans_gt, self.ref_map_node.quat_gt, 'xyzw')
				T_w_obs = T_w_mapnode @ T_mapnode_obs
				rospy.logwarn(f'[Succ] sufficient number {num_solver_inliers} solver inliers')
		
				return {'succ': True, 'T_w_obs': T_w_obs, 'solver_inliers': num_solver_inliers}
		except Exception as e:
			rospy.logwarn(f'[Fail] to estimate pose with error:', e)
			return result_fail

	def publish_message(self):
		header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id_map)
		tf_msg = ros_msg.convert_vec_to_rostf(
			np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, f"{self.frame_id_map}_graph"
		)
		self.br.sendTransform(tf_msg)
		header.frame_id += '_graph'
		ros_vis.publish_graph(
			self.image_graph, header, self.pub_graph, self.pub_graph_poses
		)

		if self.curr_obs_node is not None:
			header = Header(stamp=rospy.Time.from_sec(self.curr_obs_node.time), frame_id=self.frame_id_map)
			
			# Publish odometry and path if the local position is available
			if self.has_local_pos:
				odom = ros_msg.convert_vec_to_rosodom(self.curr_obs_node.trans, self.curr_obs_node.quat, header, self.child_frame_id)
				self.pub_odom.publish(odom)
				pose_msg = ros_msg.convert_odom_to_rospose(odom)
				
				self.path_msg.header = header
				self.path_msg.poses.append(pose_msg)
				self.pub_path.publish(self.path_msg)

			if self.curr_obs_node.has_pose_gt:
				pose_msg = ros_msg.convert_vec_to_rospose(self.curr_obs_node.trans_gt, self.curr_obs_node.quat_gt, header)
				self.path_gt_msg.header = header
				self.path_gt_msg.poses.append(pose_msg)
				self.pub_path_gt.publish(self.path_gt_msg)

			if self.ref_map_node is not None and self.args.viz:
				n_viz = 10 # visualize n_viz matched keypoints
				rgb_img_ref = (np.transpose(to_numpy(self.ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				rgb_img_obs = (np.transpose(to_numpy(self.curr_obs_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				mkpts_map, num_inliers = self.ref_map_node.get_matched_kpts()
				mkpts_obs, _ = self.curr_obs_node.get_matched_kpts()
				if mkpts_map is not None and mkpts_obs is not None:
					step_size = max(1, len(mkpts_map) // n_viz)
					rgb_img_ref_bgr = cv2.cvtColor(rgb_img_ref, cv2.COLOR_RGB2BGR)
					rgb_img_obs_bgr = cv2.cvtColor(rgb_img_obs, cv2.COLOR_RGB2BGR)
					merged_img = np.hstack((rgb_img_ref_bgr, rgb_img_obs_bgr))
					for i in range(0, len(mkpts_map), step_size):
						x0, y0 = mkpts_map[i]
						x1, y1 = mkpts_obs[i]
						cv2.circle(rgb_img_ref_bgr, (int(x0), int(y0)), 3, (0, 255, 0), -1)
						cv2.circle(rgb_img_obs_bgr, (int(x1), int(y1)), 3, (0, 255, 0), -1)
						cv2.line(merged_img, (int(x0), int(y0)), (int(x1) + rgb_img_ref.shape[1], int(y1)), (0, 255, 0), 2)	
					text = f'Matched inliers kpts: {num_inliers}'
					text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
					text_x = (merged_img.shape[1] - text_size[0])
					text_y = (merged_img.shape[0] - text_size[1])
					cv2.putText(merged_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
					img_msg = ros_msg.convert_cvimg_to_rosimg(merged_img, "bgr8", header, compressed=False)
					self.pub_map_obs.publish(img_msg)


# Test loc_pipeline without using ROS
def perform_localization(loc: LocPipeline, args):
	"""Main loop for processing observations"""
	poses = read_poses(os.path.join(args.query_data_path, 'poses.txt'))
	intrs = read_intrinsics(os.path.join(args.query_data_path, 'intrinsics.txt'))
	descs = read_descriptors(os.path.join(args.query_data_path, 'database_descriptors.txt'))
	resize = args.image_size
	
	loc.last_obs_node = None
	for node_id, (rgb_img_name, pose) in enumerate(poses.items()):
		if rospy.is_shutdown(): break	
		print(f"Loading observation {rgb_img_name}")

		rgb_img_path = os.path.join(args.query_data_path, rgb_img_name)
		rgb_img = load_rgb_image(rgb_img_path, resize)

		depth_img_name = rgb_img_name.replace('color.jpg', 'depth.png')
		depth_img_path = os.path.join(args.query_data_path, depth_img_name)
		depth_img = load_depth_image(depth_img_path, depth_scale=0.001)

		intr = intrs[rgb_img_name]
		width, height = int(intr[4]), int(intr[5])
		raw_K = np.array([intr[0], 0, intr[2], 0, intr[1], intr[3], 0, 0, 1], dtype=np.float32).reshape(3, 3)
		raw_size = (width, height)
		if resize is not None:
			K = correct_intrinsic_scale(raw_K, resize[0] / width, resize[1] / height) 
			img_size = np.array([int(resize[0]), int(resize[1])])
		else:
			K = raw_K
			img_size = raw_size

		# Create observation node
		obs_node = ImageNode(
			node_id, rgb_img, depth_img, descs[rgb_img_name],
			rospy.Time.now().to_sec(),
			np.zeros(3), np.array([0, 0, 0, 1]),
			K, img_size,
			rgb_img_name, depth_img_name
		)
		obs_node.set_raw_intrinsics(raw_K, raw_size)
		trans, quat = convert_pose_inv(
			pose[4:], 
			np.roll(pose[:4], -1), 
			'xyzw'
		)
		obs_node.set_pose_gt(trans, quat)

		loc.curr_obs_node = obs_node

		"""Perform global localization via. visual place recognition"""
		if not loc.has_global_pos:
			loc_start_time = time.time()
			result = loc.perform_global_loc(save_viz=(args.num_preds_to_save!=0))
			rospy.loginfo(f"Global localization costs: {time.time() - loc_start_time:.3f}s")
			if result['succ']:
				matched_map_id = result['map_id']
				loc.has_global_pos = True
				loc.ref_map_node = loc.image_graph.get_node(matched_map_id)
				loc.curr_obs_node.set_pose(loc.ref_map_node.trans, loc.ref_map_node.quat)
				rospy.logwarn(f'Found VPR Node in global position: {matched_map_id}')
			else:
				rospy.logwarn('[Fail] to determine the global position since no VPR results.')
				continue
		else:
			if loc.last_obs_node is not None:
				init_trans, init_quat = loc.last_obs_node.trans, loc.last_obs_node.quat
				loc.curr_obs_node.set_pose(init_trans, init_quat)

				dis_trans, _ = compute_pose_error(
					(init_trans, init_quat), 
					(loc.ref_map_node.trans, loc.ref_map_node.quat), 
					mode='vector'
				)
				if dis_trans > loc.args.global_pos_threshold:
					rospy.logwarn('Too far distance from the ref_map_node. Losing Visual Tracking. Reset the global position.')
					loc.has_global_pos = False
					loc.ref_map_node = None
			else:
				rospy.logwarn('[Fail] to determine the global position since not correct VPR.')
				continue				

		"""Perform local localization via. image matching"""
		if loc.has_global_pos:
			loc_start_time = time.time()
			result = loc.perform_local_loc()
			rospy.loginfo(f"Local localization costs: {time.time() - loc_start_time:.3f}s")
			if result['succ']:
				T_w_obs = result['T_w_obs']
				trans, quat = convert_matrix_to_vec(T_w_obs, 'xyzw')
				loc.curr_obs_node.set_pose(trans, quat)
				loc.has_local_pos = True
				rospy.loginfo(f'Groundtruth Poses: {loc.curr_obs_node.trans_gt.T}')
				rospy.loginfo(f'Estimated Poses: {trans.T}\n')
			else:
				loc.has_local_pos = False
				rospy.logwarn('[Fail] to determine the local position.')

		loc.publish_message()
		# Set as the initial guess of the next observation
		loc.last_obs_node = loc.curr_obs_node
		time.sleep(0.01)

		input()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.map_path, 'output_loc_pipeline'))

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, out_dir)
	rospy.loginfo('Initialize VPR Model')
	loc_pipeline.init_vpr_model()
	rospy.loginfo('Initialize Image Matcher')
	loc_pipeline.init_img_matcher()
	rospy.loginfo('Initialize Pose Solver')
	loc_pipeline.init_pose_solver()
	loc_pipeline.read_covis_graph_from_files()

	rospy.init_node('loc_pipeline_node', anonymous=True)
	loc_pipeline.initalize_ros()
	loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'vloc_map')
	loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_localization(loc_pipeline, args)
