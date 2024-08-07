#!/usr/bin/env python

"""
Usage: 
python loc_pipeline.py \
--dataset_path /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_17DRP5sb8fy/out_map \
--image_size 288 512 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors --num_preds_to_save 3 \
--img_matcher master --save_img_matcher \
--pose_solver pnp --config_pose_solver config/dataset/matterport3d.yaml \
--no_viz

Usage: 
rosbag record -O /Titan/dataset/data_topo_loc/anymal_lab_upstair_20240722_0/vloc.bag \
/vloc/odom /vloc/path /vloc/path_gt /vloc/image_map_obs
"""

import os
import sys

import pathlib
import numpy as np
import torch
import time
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros

from matching.utils import to_numpy
from utils.utils_vpr_method import initialize_vpr_model, perform_knn_search
from utils.utils_vpr_method import save_visualization as save_vpr_visualization
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image import load_rgb_image, load_depth_image
from utils.utils_pipeline import *
from utils.pose_solver import get_solver
from utils.pose_solver_default import cfg
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

class LocPipeline:
	def __init__(self, args, log_dir):
		self.args = args
		self.log_dir = log_dir
		self.has_global_pos = False
		self.has_local_position = False

		self.vpr_model = initialize_vpr_model(self.args.vpr_method, self.args.vpr_backbone, self.args.vpr_descriptors_dimension, self.args.device)
		logging.info(f"VPR model: {self.args.vpr_method}")

		self.img_matcher = initialize_img_matcher(self.args.img_matcher, self.args.device, self.args.n_kpts)
		logging.info(f"Image matcher: {self.args.img_matcher}")
		
		cfg.merge_from_file(self.args.config_pose_solver)
		self.pose_solver = get_solver(self.args.pose_solver, cfg)
		logging.info(f"Pose solver: {self.args.pose_solver}")

		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		
		self.pub_odom = rospy.Publisher('/vloc/odom', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/vloc/path', Path, queue_size=10)
		self.pub_path_gt = rospy.Publisher('/vloc/path_gt', Path, queue_size=10)
		self.pub_map_obs = rospy.Publisher('/vloc/image_map_obs', Image, queue_size=10)

		self.br = tf2_ros.TransformBroadcaster()
		self.path_msg = Path()
		self.path_gt_msg = Path()

	def read_map_from_file(self):
		data_path = self.args.dataset_path
		self.image_graph = GraphLoader.load_data(
			data_path,
			self.args.image_size,
			depth_scale=self.args.depth_scale,
			normalized=False
		)
		logging.info(f"Loaded {self.image_graph} from {data_path}")

		# Extract VPR descriptors for all nodes in the map
		# Constant variables
		self.DB_DESCRIPTORS_ID = np.array(self.image_graph.get_all_id())
		self.DB_DESCRIPTORS = np.array([map_node.get_descriptor() for _, map_node in self.image_graph.nodes.items()], dtype="float32")
		print(f"IDs: {self.DB_DESCRIPTORS_ID} extracted {self.DB_DESCRIPTORS.shape} VPR descriptors.")

		self.DB_POSES = np.empty((self.image_graph.get_num_node(), 7), dtype="float32")
		for indices, (_, map_node) in enumerate(self.image_graph.nodes.items()):
			self.DB_POSES[indices, :3] = map_node.trans
			self.DB_POSES[indices, 3:] = map_node.quat

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

	def perform_image_matching(self, map_node, obs_node):
		try:
			# obs_node.rgb_image has depth
			matcher_result = self.img_matcher(obs_node.rgb_image, map_node.rgb_image)
			num_inliers, H, mkpts0, mkpts1 = (
				matcher_result["num_inliers"],
				matcher_result["H"],
				matcher_result["inliers0"],
				matcher_result["inliers1"],
			)
			out_str = f"Paths: map_id ({map_node.id}), obs_id ({obs_node.id}). "
			out_str += f"Found {num_inliers} inliers after RANSAC. "

			"""Save matching results"""
			if self.args.save_img_matcher:
				save_img_matcher_visualization(
					obs_node.rgb_image, map_node.rgb_image,
					mkpts0, mkpts1, self.log_dir, obs_node.id, n_viz=100)
				
			return matcher_result
		except Exception as e:
			logging.error(f"Error in image matching: {e}")
		return None

	def publish_message(self):
		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id = 'map'

		tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, 'map_graph')
		self.br.sendTransform(tf_msg)
		header.frame_id = 'map_graph'
		pytool_ros.ros_vis.publish_graph(self.image_graph, header, self.pub_graph, self.pub_graph_poses)

		if self.curr_obs_node is not None:
			header.frame_id = "map"
			child_frame_id = "camera"
			odom_msg = pytool_ros.ros_msg.convert_vec_to_rosodom(self.curr_obs_node.trans, self.curr_obs_node.quat, header, child_frame_id)
			self.pub_odom.publish(odom_msg)

			pose_msg = pytool_ros.ros_msg.convert_odom_to_rospose(odom_msg)
			self.path_msg.header = header
			self.path_msg.poses.append(pose_msg)
			self.pub_path.publish(self.path_msg)

			tf_msg = pytool_ros.ros_msg.convert_odom_to_rostf(odom_msg)
			self.br.sendTransform(tf_msg)

			if self.curr_obs_node.has_pose_gt:
				pose_msg = pytool_ros.ros_msg.convert_vec_to_rospose(self.curr_obs_node.trans_gt, self.curr_obs_node.quat_gt, header)
				self.path_gt_msg.header = header
				self.path_gt_msg.poses.append(pose_msg)
				self.pub_path_gt.publish(self.path_gt_msg)

			if self.ref_map_node is not None:
				rgb_img_map_node = (np.transpose(to_numpy(self.ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				rgb_img_obs = (np.transpose(to_numpy(self.curr_obs_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				rgb_img_merge = np.hstack((rgb_img_map_node, rgb_img_obs))
				img_msg = pytool_ros.ros_msg.convert_cvimg_to_rosimg(rgb_img_merge, "rgb8", header, compressed=False)
				self.pub_map_obs.publish(img_msg)

	def run(self):
		rospy.init_node('loc_pipeline_node', anonymous=True)

		"""Main loop for processing observations"""
		obs_poses_gt = np.loadtxt(os.path.join(self.args.dataset_path, '../out_general', 'poses.txt'))
		obs_cam_intrinsics = np.loadtxt(os.path.join(self.args.dataset_path, '../out_general', 'intrinsics.txt'))

		rate = rospy.Rate(100)
		for obs_id in range(100, len(obs_poses_gt), 10):
			if rospy.is_shutdown(): break

			# Load observation data
			print(f"Loading observation with id {obs_id}")
			img_size = self.args.image_size

			rgb_img_path = os.path.join(self.args.dataset_path, '../out_general/seq', f'{obs_id:06d}.color.jpg')
			rgb_img = load_rgb_image(rgb_img_path, img_size, normalized=False)

			depth_img_path = os.path.join(self.args.dataset_path, '../out_general/seq', f'{obs_id:06d}.depth.png')
			depth_img = load_depth_image(depth_img_path, img_size, depth_scale=self.args.depth_scale)

			K = np.array([obs_cam_intrinsics[obs_id, 0], 0, obs_cam_intrinsics[obs_id, 2], 0, 
						  obs_cam_intrinsics[obs_id, 1], obs_cam_intrinsics[obs_id, 3], 
						  0, 0, 1], dtype=np.float32).reshape(3, 3)
			raw_img_size = (obs_cam_intrinsics[obs_id, 4], obs_cam_intrinsics[obs_id, 5]) # width, height
			if img_size is not None:
				K = pytool_sensor.utils.correct_intrinsic_scale(K, img_size[0] / raw_img_size[0], img_size[1] / raw_img_size[1])
			else:
				img_size = raw_img_size

			# Extract VPR descriptors
			vpr_start_time = time.time()
			with torch.no_grad():
				desc = self.vpr_model(rgb_img.unsqueeze(0).to(self.args.device)).cpu().numpy()
			print(f"Extract VPR descriptors cost: {time.time() - vpr_start_time:.3f}s")

			# Create observation node
			obs_node = ImageNode(obs_id, rgb_img, depth_img, desc,
								 0, np.zeros(3), np.array([0, 0, 0, 1]),
								 K, img_size,
								 rgb_img_path, depth_img_path)
			obs_node.set_pose_gt(obs_poses_gt[obs_id, 1:4], obs_poses_gt[obs_id, 4:])
			self.curr_obs_node = obs_node

			"""Perform global localization via. visual place recognition"""
			if not self.has_global_pos:
				gl_start_time = time.time()
				vpr_dis, vpr_pred = self.perform_vpr(self.DB_DESCRIPTORS, self.curr_obs_node.get_descriptor())
				vpr_dis, vpr_pred = vpr_dis[0, :], vpr_pred[0, :]
				print(f"Global localization time via. VPR: {time.time() - gl_start_time:.3f}s")
				if len(vpr_pred) == 0:
					print('No start node found, cannot determine the global position.')
					continue

				# Save VPR visualization for the top-k predictions
				if self.args.num_preds_to_save != 0:
					list_of_images_paths = [self.curr_obs_node.rgb_img_path]
					for i in range(len(vpr_pred[:self.args.num_preds_to_save])):
						map_node = self.image_graph.get_node(self.DB_DESCRIPTORS_ID[vpr_pred[i]])
						list_of_images_paths.append(map_node.rgb_img_path)
					preds_correct = [None] * len(list_of_images_paths)
					save_vpr_visualization(self.log_dir, 0, list_of_images_paths, preds_correct)				

				self.has_global_pos = True
				self.global_pos_node = self.image_graph.get_node(self.DB_DESCRIPTORS_ID[vpr_pred[0]])
				self.curr_obs_node.set_pose(self.global_pos_node.trans, self.global_pos_node.quat)
				self.last_obs_node = None
			else:
				init_trans, init_quat = self.last_obs_node.trans, self.last_obs_node.quat
				self.curr_obs_node.set_pose(init_trans, init_quat)

			"""Perform local localization via. image matching"""
			if self.has_global_pos:
				# _, knn_pred = perform_knn_search(self.DB_POSES[:, :3], self.curr_obs_node.trans.reshape(1, -1), 3, recall_values=[5])
				# knn_pred = knn_pred[0]
				# if len(knn_pred) == 0: continue
				# dis_TF_list = [pytool_math.tools_eigen.compute_relative_dis_TF(
				# 	pytool_math.tools_eigen.convert_vec_to_matrix(self.curr_obs_node.trans, self.curr_obs_node.quat, 'xyzw'),
				# 	pytool_math.tools_eigen.convert_vec_to_matrix(self.DB_POSES[knn_pred[i], :3], self.DB_POSES[knn_pred[i], 3:], 'xyzw')
				# ) for i in range(len(knn_pred))]

				#######################################################
				# DEBUG(gogojjh): may fin dwrong reference node
				#######################################################
				min_dis = 10.0
				knn_dis, knn_pred = perform_knn_search(self.DB_POSES[:, :3], self.curr_obs_node.trans.reshape(1, -1), 3, recall_values=[5])
				knn_dis, knn_pred = knn_dis[0], knn_pred[0]
				knn_pred, knn_dis = knn_pred[knn_dis < min_dis], knn_dis[knn_dis < min_dis]
				db_descriptors_select = self.DB_DESCRIPTORS[knn_pred, :]
				db_descriptors_id_select = self.DB_DESCRIPTORS_ID[knn_pred]
				print('db_descriptors_id_select: ', db_descriptors_id_select)
				vpr_dis, vpr_pred = self.perform_vpr(db_descriptors_select, self.curr_obs_node.get_descriptor())
				vpr_dis, vpr_pred = vpr_dis[0], vpr_pred[0]
				while len(vpr_pred) > 0:
					map_id = db_descriptors_id_select[vpr_pred[0]]
					self.ref_map_node = self.image_graph.get_node(map_id)
					print(f'Found the reference map node: {self.ref_map_node.id}')
	
					im_start_time = time.time()
					matcher_result = self.perform_image_matching(self.ref_map_node, self.curr_obs_node)
					print(f"Local localization time via. Image Matching: {time.time() - im_start_time:.3f}s")
	
					if matcher_result is None or matcher_result["num_inliers"] < 100:					
						vpr_pred = np.delete(vpr_pred, 0)
						continue
					else:
						break
				if len(vpr_pred) == 0: continue
				try:
					T_mapnode_obs = None
					if self.args.img_matcher == "mickey":
						R, t = self.img_matcher.scene["R"].squeeze(0), self.img_matcher.scene["t"].squeeze(0)
						R, t = to_numpy(R), to_numpy(t)
						T_mapnode_obs = np.eye(4)
						T_mapnode_obs[:3, :3], T_mapnode_obs[:3, 3] = R, t
						print(f'Mickey Solver:\n', T_mapnode_obs)
					else:
						depth_img0 = to_numpy(self.curr_obs_node.depth_image.squeeze(0))
						mkpts0, mkpts1 = (
							matcher_result["inliers0"],
							matcher_result["inliers1"],
						)					
						R, t, inliers = self.pose_solver.estimate_pose(
							mkpts0, mkpts1, 
							self.curr_obs_node.K, self.ref_map_node.K, 
							depth_img0, None)
						T_mapnode_obs = np.eye(4)
						T_mapnode_obs[:3, :3], T_mapnode_obs[:3, 3] = R, t.reshape(3)
						print(f'{self.args.pose_solver}: Number of inliers: {inliers}\n', T_mapnode_obs)

					if T_mapnode_obs is not None:
						T_w_mapnode = pytool_math.tools_eigen.convert_vec_to_matrix(
							self.ref_map_node.trans_gt, self.ref_map_node.quat_gt, 'xyzw')
						T_w_obs = T_w_mapnode @ T_mapnode_obs
						trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(T_w_obs, 'xyzw')
						self.curr_obs_node.set_pose(trans, quat)
						print(f'Groundtruth Poses: {self.curr_obs_node.trans_gt.T}')
						print(f'Estimated Poses: {trans.T}\n')
				except Exception as e:
					print(f'Failed to estimate pose with {self.args.pose_solver}:', e)

			self.publish_message()
			self.last_obs_node = self.curr_obs_node
			rate.sleep()
			# input()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_loc_pipeline'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	loc_pipeline = LocPipeline(args, log_dir)
	loc_pipeline.read_map_from_file()
	loc_pipeline.run()
