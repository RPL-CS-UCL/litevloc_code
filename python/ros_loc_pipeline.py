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
	# Keep the queue size to be small for processing the newest data
	while rgb_depth_queue.qsize() > 1: rgb_depth_queue.get()
	lock.release()

def odom_callback(odom_msg):
	time = odom_msg.header.stamp.to_sec()
	trans, quat = pytool_ros.ros_msg.convert_rosodom_to_vec(odom_msg)
	T = pytool_math.tools_eigen.convert_vec_to_matrix(trans, quat)
	fused_poses.add(time, T)

def perform_localization(loc: LocPipeline, args):
	obs_id = 0
	min_depth, max_depth = 0.1, 15.0
	while not rospy.is_shutdown():
		if not rgb_depth_queue.empty():
			lock.acquire()
			rgb_img_msg, depth_img_msg, camera_info_msg = rgb_depth_queue.get()
			lock.release()
			loc.child_frame_id = rgb_img_msg.header.frame_id
			rgb_img_time = rgb_img_msg.header.stamp.to_sec()
			rgb_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(rgb_img_msg)
			depth_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(depth_img_msg)
			if depth_img_msg.encoding == "mono16": depth_img *= 0.001
			depth_img[(depth_img < min_depth) | (depth_img > max_depth)] = 0.0
			raw_K = np.array(camera_info_msg.K).reshape((3, 3))
			raw_img_size = (camera_info_msg.width, camera_info_msg.height)

			img_size = args.image_size
			rgb_img_tensor = rgb_image_to_tensor(rgb_img, img_size, normalized=False)
			depth_img_tensor = depth_image_to_tensor(depth_img, img_size, depth_scale=1.0)
			if img_size is not None:
				K = pytool_sensor.utils.correct_intrinsic_scale(
					raw_K, 
					img_size[0] / raw_img_size[0], 
					img_size[1] / raw_img_size[1])
			else:
				K = raw_K
				img_size = raw_img_size

			"""Process the current observation"""
			# TODO(gogojjh):
			vpr_start_time = time.time()
			with torch.no_grad():
				desc = loc.vpr_model(rgb_img_tensor.unsqueeze(0).to(args.device)).cpu().numpy()
			print(f"Extract VPR descriptors cost: {time.time() - vpr_start_time:.3f}s")
			
			obs_node = ImageNode(obs_id, rgb_img_tensor, depth_img_tensor, desc,
								 rgb_img_time, np.zeros(3), np.array([0, 0, 0, 1]),
								 K, img_size, '', '')
			loc.curr_obs_node = obs_node

			"""Perform global localization via. visual place recognition"""
			if not loc.has_global_pos:
				loc_start_time = time.time()
				result = loc.perform_global_loc(save=False)
				print(f"Global localization cost: {time.time() - loc_start_time:.3f}s")
				if result['succ']:
					matched_map_id = result['map_id']
					loc.has_global_pos = True
					loc.global_pos_node = loc.image_graph.get_node(matched_map_id)
					loc.curr_obs_node.set_pose(loc.global_pos_node.trans, loc.global_pos_node.quat)
					loc.last_obs_node = None
				else:
					print('Failed to determine the global position.')
					continue
			else:
				# Initialize the current transformation using the historical fused poses
				idx_closest, stamped_pose_closest = fused_poses.find_closest(obs_node.time)
				if idx_closest is None:
					init_trans, init_quat = loc.last_obs_node.trans, loc.last_obs_node.quat
				else:
					# print("Given initial pose: ", stamped_pose_closest[1])
					init_trans, init_quat = pytool_math.tools_eigen.convert_matrix_to_vec(stamped_pose_closest[1])
				loc.curr_obs_node.set_pose(init_trans, init_quat)

			"""Perform local localization via. image matching"""
			if loc.has_global_pos:
				loc_start_time = time.time()
				result = loc.perform_local_pos()
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
			loc.last_obs_node = loc.curr_obs_node

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

	# Subscribe to RGB, depth images, and odometry
	if args.ros_rgb_img_type == 'raw':
		rgb_sub = message_filters.Subscriber('/color/image', Image)
	else:
		rgb_sub = message_filters.Subscriber('/color/image', CompressedImage)
	depth_sub = message_filters.Subscriber('/depth/image', Image)
	camera_info_sub = message_filters.Subscriber('/color/camera_info', CameraInfo)
	# Synchronize the topics
	ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, camera_info_sub], queue_size=10, slop=0.1)
	ts.registerCallback(rgb_depth_image_callback)

	# Subscribe to fusion odometry
	fusion_odom_sub = rospy.Subscriber('/pose_fusion/odometry', Odometry, odom_callback)

	# Start the localization thread
	localization_thread = threading.Thread(target=perform_localization, args=(loc_pipeline, args, ))
	localization_thread.start()

	rospy.spin()