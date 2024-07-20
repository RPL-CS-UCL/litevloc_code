#!/usr/bin/env python

"""
Usage: python loc_pipeline.py --dataset_path /Titan/dataset/data_topo_loc/anymal_ops_mos --image_size 288 512 --device=cuda \
--sample_map 1 --sample_obs 5 --depth_scale 0.001 --min_depth_pro 0.1 --max_depth_pro 5.5 \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors --num_preds_to_save 3 \
--img_matcher duster --save_img_matcher --no_viz 
"""

import os
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
from utils.utils_image_matching_method import initialize_img_matcher, compute_scale_factor, plot_images
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image_matching_method import save_output as save_img_matcher_output
from utils.utils_image import load_rgb_image, load_depth_image
from utils.utils_pipeline import *
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode

import pycpptools.src.python.utils_algorithm as pytool_alg
import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros

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
		self.img_matcher = initialize_img_matcher(self.args.img_matcher, self.args.device, self.args.n_kpts)

		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		
		self.pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/path', Path, queue_size=10)
		self.pub_path_gt = rospy.Publisher('/path_gt', Path, queue_size=10)
		self.pub_map_obs = rospy.Publisher('/image_map_obs', Image, queue_size=10)

		self.br = tf2_ros.TransformBroadcaster()
		self.path_msg = Path()
		self.path_gt_msg = Path()

	def read_map_from_file(self):
		data_path = os.path.join(self.args.dataset_path, 'map')
		self.image_graph = GraphLoader.load_data(
			data_path,
			self.args.image_size,
			self.args.depth_scale,
			normalized=False
		)
		logging.info(f"Loaded {self.image_graph} from {data_path}")

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
			matcher_result = self.img_matcher(map_node.rgb_image, obs_node.rgb_image)
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
				save_img_matcher_visualization(map_node.rgb_image, obs_node.rgb_image, 
																			 mkpts0, mkpts1, self.log_dir, obs_node.id, n_viz=100)
				# save_img_matcher_output(matcher_result, None, None, 
				# 												self.args.img_matcher, self.args.n_kpts, 
				# 												self.args.image_size, self.log_dir, obs_node.id)
			if num_inliers > 100:
				depth_img_meas = np.squeeze(np.transpose(to_numpy(obs_node.depth_image), (1, 2, 0)), axis=2)
				depth_img_est = to_numpy(self.img_matcher.scene.get_depthmaps())[1]
				mask = (depth_img_meas < self.args.min_depth_pro) | (depth_img_meas > self.args.max_depth_pro)
				depth_img_meas[mask] = 0.0
				depth_img_est[mask] = 0.0
				meas_scale = compute_scale_factor(depth_img_meas, depth_img_est)
				# plot_images(depth_img_meas, depth_img_est * meas_scale, title1='Depth (GT)', title2='Depth (Est)', 
				# 						save_path=os.path.join(self.log_dir, 'preds', f'depth_map_{obs_node.id:06d}.jpg'))
				im_poses = to_numpy(self.img_matcher.scene.get_im_poses())
				est_T_ref_obs = np.linalg.inv(im_poses[0]) if abs(np.sum(np.diag(im_poses[1])) - 4.0) < 1e-5 else im_poses[1]
				est_T_ref_obs[:3, 3] *= meas_scale
				matcher_result["meas_scale"] = meas_scale
				matcher_result["est_T_ref_obs"] = est_T_ref_obs
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

		# Extract VPR descriptors for all nodes in the map
		db_descriptors_id = np.array(self.image_graph.get_all_id())
		db_descriptors = np.array([map_node.get_descriptor() for _, map_node in self.image_graph.nodes.items()], dtype="float32")
		print(f"IDs: {db_descriptors_id} extracted {db_descriptors.shape} VPR descriptors.")

		"""Main loop for processing observations"""
		obs_poses_gt = np.loadtxt(os.path.join(self.args.dataset_path, 'obs', 'camera_pose_gt.txt'))

		rate = rospy.Rate(100)
		for obs_id in range(0, len(obs_poses_gt), 15):
			if rospy.is_shutdown(): 
				break

			# Load observation data
			print(f"obs_id: {obs_id}")
			rgb_img_path = os.path.join(self.args.dataset_path, 'obs/rgb', f'{obs_id:06d}.png')
			rgb_img = load_rgb_image(rgb_img_path, self.args.image_size, normalized=False)
			depth_img_path = os.path.join(self.args.dataset_path, 'obs/depth', f'{obs_id:06d}.png')
			depth_img = load_depth_image(depth_img_path, self.args.image_size, depth_scale=self.args.depth_scale)
			with torch.no_grad():
				desc = self.vpr_model(rgb_img.unsqueeze(0).to(self.args.device)).cpu().numpy()
			obs_node = ImageNode(obs_id, rgb_img, depth_img, desc, 0,
													 np.zeros(3), np.array([0, 0, 0, 1]),
													 rgb_img_path, depth_img_path)
			obs_node.set_pose_gt(obs_poses_gt[obs_id, 1:4], obs_poses_gt[obs_id, 4:])
			self.curr_obs_node = obs_node

			"""Perform global localization via. visual place recognition"""
			if not self.has_global_pos:
				vpr_start_time = time.time()
				vpr_dis, vpr_pred = self.perform_vpr(db_descriptors, self.curr_obs_node.get_descriptor())
				vpr_dis, vpr_pred = vpr_dis[0, :], vpr_pred[0, :]
				print(f"Global localization time via. VPR: {time.time() - vpr_start_time:.3f}s")
				if len(vpr_pred) == 0:
					print('No start node found, cannot determine the global position.')
					continue

				# Save VPR visualization for the top-k predictions
				if self.args.num_preds_to_save != 0:
					list_of_images_paths = [self.curr_obs_node.rgb_img_path]
					for i in range(len(vpr_pred[:self.args.num_preds_to_save])):
						map_node = self.image_graph.get_node(db_descriptors_id[vpr_pred[i]])
						list_of_images_paths.append(map_node.rgb_img_path)
					preds_correct = [None] * len(list_of_images_paths)
					save_vpr_visualization(self.log_dir, 0, list_of_images_paths, preds_correct)				

				self.has_global_pos = True
				self.global_pos_node = self.image_graph.get_node(db_descriptors_id[vpr_pred[0]])
				self.curr_obs_node.set_pose(self.global_pos_node.trans, self.global_pos_node.quat)
				self.last_obs_node = None
			else:
				init_trans, init_quat = self.last_obs_node.trans, self.last_obs_node.quat
				self.curr_obs_node.set_pose(init_trans, init_quat)

			"""Perform local localization via. image matching"""
			if self.has_global_pos:
				db_poses = np.empty((self.image_graph.get_num_node(), 3), dtype="float32")
				for indices, (_, map_node) in enumerate(self.image_graph.nodes.items()):
					db_poses[indices, :] = map_node.trans
				query_pose = self.curr_obs_node.trans.reshape(1, 3)

				min_dis = 25.0
				knn_dis, knn_pred = perform_knn_search(db_poses, query_pose, 3, recall_values=[10])
				knn_dis, knn_pred = knn_dis[0], knn_pred[0]
				knn_pred, knn_dis = knn_pred[knn_dis < min_dis], knn_dis[knn_dis < min_dis]
				db_descriptors_select = db_descriptors[knn_pred, :]
				db_descriptors_id_select = db_descriptors_id[knn_pred]
				print('db_descriptors_id_select: ', db_descriptors_id_select)

				vpr_dis, vpr_pred = self.perform_vpr(db_descriptors_select, self.curr_obs_node.get_descriptor())
				self.ref_map_node = self.image_graph.get_node(db_descriptors_id_select[vpr_pred[0, 0]])
				# print(f'VPR dis: {vpr_dis}')
				print(f'Found the reference map node: {self.ref_map_node.id}')

				im_start_time = time.time()
				matcher_result = self.perform_image_matching(self.ref_map_node, self.curr_obs_node)
				print(f"Local localization time via. Image Matching: {time.time() - im_start_time:.3f}s")

				if matcher_result is not None:
					print(f'Groundtruth Poses: {self.curr_obs_node.trans_gt.T}')
					meas_scale, est_T_ref_obs = matcher_result["meas_scale"], matcher_result["est_T_ref_obs"]
					T_w_map_node = pytool_math.tools_eigen.convert_vec_to_matrix(
						self.ref_map_node.trans_gt, 
						self.ref_map_node.quat_gt, 
						'xyzw'
					)
					T_w_obs = T_w_map_node @ est_T_ref_obs
					trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(T_w_obs, 'xyzw')
					self.curr_obs_node.set_pose(trans, quat)
					print(f'Estimated Poses with Meas scale {meas_scale:.3f}: {trans.T}\n')

					if not self.args.no_viz:
						self.img_matcher.scene.show(cam_size=0.05)				

			self.publish_message()
			self.last_obs_node = self.curr_obs_node
			rate.sleep()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_loc_pipeline'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	loc_pipeline = LocPipeline(args, log_dir)
	loc_pipeline.read_map_from_file()
	loc_pipeline.run()
