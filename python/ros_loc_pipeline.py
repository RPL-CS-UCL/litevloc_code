#! /opt/conda/envs/topo_loc/bin/python

"""
Usage: 
python loc_pipeline.py \
--dataset_path /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_17DRP5sb8fy/out_map \
--image_size 288 512 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors --num_preds_to_save 3 \
--img_matcher master --save_img_matcher \
--pose_solver pnp --config_pose_solver config/dataset/matterport3d.yaml \
--viz

Usage: 
rosbag record -O /Titan/dataset/data_topo_loc/anymal_lab_upstair_20240722_0/vloc.bag \
/vloc/odom /vloc/path /vloc/path_gt /vloc/image_map_obs
"""

# General
import os
import sys

# ROS
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters

import pathlib
import numpy as np
import torch
import time
import queue
import threading

from utils.utils_image import rgb_image_to_tensor, depth_image_to_tensor
from utils.utils_pipeline import *
from image_node import ImageNode
from loc_pipeline import LocPipeline

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor
import pycpptools.src.python.utils_algorithm as pytool_algo

# if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

fused_poses = pytool_algo.stpose.StampedPoses()
rgb_depth_queue = queue.Queue()
lock = threading.Lock()

def rgb_depth_image_callback(rgb_img_msg, depth_img_msg, camera_info_msg):
	lock.acquire()
	rgb_depth_queue.put((rgb_img_msg, depth_img_msg, camera_info_msg))
	while rgb_depth_queue.qsize() > 1: rgb_depth_queue.get()
	lock.release()

def odom_callback(odom_msg):
	time = odom_msg.header.stamp.to_sec()
	trans, quat = pytool_ros.ros_msg.convert_rosodom_to_vec(odom_msg)
	T = pytool_math.tools_eigen.convert_vec_to_matrix(trans, quat)
	fused_poses.add(time, T)

def perform_localization(loc: LocPipeline, args):
	obs_id = 0
	resize = args.image_size
	r = rospy.Rate(loc.main_freq)
	while not rospy.is_shutdown():
		if not rgb_depth_queue.empty():
			"""Get the latest RGB, depth images, and camera info"""
			lock.acquire()
			rgb_img_msg, depth_img_msg, camera_info_msg = rgb_depth_queue.get()
			lock.release()
			loc.child_frame_id = rgb_img_msg.header.frame_id
			rgb_img_time = rgb_img_msg.header.stamp.to_sec()
			rgb_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(rgb_img_msg)
			depth_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(depth_img_msg)
			if depth_img_msg.encoding == "mono16": depth_img *= 0.001
			depth_img[(depth_img < loc.depth_range[0]) | (depth_img > loc.depth_range[1])] = 0.0
			# To tensor
			rgb_img_tensor = rgb_image_to_tensor(rgb_img, resize, normalized=False)
			depth_img_tensor = depth_image_to_tensor(depth_img, depth_scale=1.0)
			# Intrinsic matrix
			raw_K = np.array(camera_info_msg.K).reshape((3, 3))
			raw_img_size = (camera_info_msg.width, camera_info_msg.height)
			K = pytool_sensor.utils.correct_intrinsic_scale(raw_K, resize[0] / raw_img_size[0], resize[1] / raw_img_size[1]) if resize is not None else raw_K
			img_size = (int(resize[0]), int(resize[1])) if resize is not None else raw_img_size

			# Create observation node
			with torch.no_grad():
				desc = loc.vpr_model(rgb_img_tensor.unsqueeze(0).to(args.device)).cpu().numpy()
			obs_node = ImageNode(obs_id, rgb_img_tensor, depth_img_tensor, desc,
								 rgb_img_time, np.zeros(3), np.array([0, 0, 0, 1]),
								 K, img_size, 
								 '', '')
			obs_node.set_raw_intrinsics(raw_K, raw_img_size)
			loc.curr_obs_node = obs_node

			"""Perform global localization via. visual place recognition"""
			if not loc.has_global_pos:
				loc_start_time = time.time()
				result = loc.perform_global_loc(save=False)
				print(f"Global localization cost: {time.time() - loc_start_time:.3f}s")
				if result['succ']:
					matched_map_id = result['map_id']
					loc.has_global_pos = True
					loc.ref_map_node = loc.image_graph.get_node(matched_map_id)
					loc.curr_obs_node.set_pose(loc.ref_map_node.trans, loc.ref_map_node.quat)
					print(f'Found VPR Node in global position: {matched_map_id}')
				else:
					print('Failed to determine the global position since no VPR results.')
					continue
			else:
				# Initialize the current transformation using the historical fused poses
				idx_closest, stamped_pose_closest = fused_poses.find_closest(loc.curr_obs_node.time)
				# No fused poses available
				if idx_closest is None:
					init_trans, init_quat = loc.ref_map_node.trans, loc.ref_map_node.quat
				# Use the closest fused pose as the initial guess
				else:
					init_trans, init_quat = pytool_math.tools_eigen.convert_matrix_to_vec(stamped_pose_closest[1])
				loc.curr_obs_node.set_pose(init_trans, init_quat)
				
				dis_trans, _ = pytool_math.tools_eigen.compute_relative_dis(init_trans, init_quat, loc.ref_map_node.trans, loc.ref_map_node.quat)
				if dis_trans > loc.args.global_pos_threshold:
					print('Too far distance from the ref_map_node. Losing Visual Tracking')
					print('Reset the global position.')
					loc.has_global_pos = False
					loc.ref_map_node = None
					continue

			"""Perform local localization via. image matching"""
			if loc.has_global_pos:
				loc_start_time = time.time()
				result = loc.perform_local_loc()
				print(f"Local localization cost: {time.time() - loc_start_time:.3f}s")
				if result['succ']:
					T_w_obs = result['T_w_obs']
					trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(T_w_obs, 'xyzw')
					loc.curr_obs_node.set_pose(trans, quat)
					print(f'Estimated Poses: {trans.T}\n')
				else:
					print('Failed to determine the local position.')
					continue
			loc.publish_message()
			r.sleep()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_ros_loc_pipeline'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, log_dir)
	loc_pipeline.init_vpr_model()
	loc_pipeline.init_img_matcher()
	loc_pipeline.init_pose_solver()
	loc_pipeline.read_map_from_file()

	rospy.init_node('ros_loc_pipeline_simu', anonymous=False)
	loc_pipeline.initalize_ros()
	loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	loc_pipeline.main_freq = rospy.get_param('~main_freq', 1)
	min_depth = rospy.get_param('~min_depth', 0.1)
	max_depth = rospy.get_param('~max_depth', 15.0)
	loc_pipeline.depth_range = (min_depth, max_depth)

	# Subscribe to RGB, depth images, and odometry
	if args.ros_rgb_img_type == 'raw':
		rgb_sub = message_filters.Subscriber('/color/image', Image)
	else:
		rgb_sub = message_filters.Subscriber('/color/image', CompressedImage)
	depth_sub = message_filters.Subscriber('/depth/image', Image)
	camera_info_sub = message_filters.Subscriber('/color/camera_info', CameraInfo)
	ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, camera_info_sub], queue_size=10, slop=0.1)
	ts.registerCallback(rgb_depth_image_callback)

	# Subscribe to fusion odometry
	fusion_odom_sub = rospy.Subscriber('/pose_fusion/odometry', Odometry, odom_callback)

	# Start the localization thread
	localization_thread = threading.Thread(target=perform_localization, args=(loc_pipeline, args, ))
	localization_thread.start()

	rospy.spin()