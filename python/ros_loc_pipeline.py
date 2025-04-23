#! /usr/bin/env python

"""
Usage: 
python python/ros_loc_pipeline.py \
	--map_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--image_size 512 288 \
	--device cuda --vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 256  \
	--img_matcher master --pose_solver pnp  \
	--config_pose_solver config/dataset/matterport3d.yaml \
	--ros_rgb_img_type raw \
	--global_pos_threshold 10.0 \
	--min_master_conf_thre 1.5 \
	--min_kpts_inliers_thre 300  \
	--min_solver_inliers_thre 300
"""

# General
import os
import sys
import pathlib
import numpy as np
import torch
import time
import queue
import threading

# ROS
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters

# Others
from utils.utils_image import rgb_image_to_tensor, depth_image_to_tensor, to_numpy
from utils.utils_pipeline import *
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, compute_pose_error, correct_intrinsic_scale
from utils.utils_ros import ros_msg
from utils.utils_stamped_poses import StampedPoses
from image_node import ImageNode
from loc_pipeline import LocPipeline

fused_poses = StampedPoses()
rgb_depth_queue = queue.Queue()
lock = threading.Lock()

def rgb_depth_image_callback(rgb_img_msg, depth_img_msg, camera_info_msg):
	lock.acquire()
	rgb_depth_queue.put((rgb_img_msg, depth_img_msg, camera_info_msg))
	while rgb_depth_queue.qsize() > 1: rgb_depth_queue.get()
	lock.release()

def odom_callback(odom_msg):
	time = odom_msg.header.stamp.to_sec()
	trans, quat = ros_msg.convert_rosodom_to_vec(odom_msg)
	T = convert_vec_to_matrix(trans, quat)
	fused_poses.add(time, T)

def perform_localization(loc: LocPipeline, args):
	obs_id = 0
	resize = args.image_size # WxH
	r = rospy.Rate(loc.main_freq)
	while not rospy.is_shutdown():
		if not rgb_depth_queue.empty():
			"""Get the latest RGB, depth images, and camera info"""
			lock.acquire()
			rgb_img_msg, depth_img_msg, camera_info_msg = rgb_depth_queue.get()
			lock.release()
			loc.child_frame_id = rgb_img_msg.header.frame_id
			rgb_img_time = rgb_img_msg.header.stamp.to_sec()
			rgb_img = ros_msg.convert_rosimg_to_cvimg(rgb_img_msg)
			depth_img = ros_msg.convert_rosimg_to_cvimg(depth_img_msg)
			if depth_img_msg.encoding == "mono16": depth_img *= 0.001
			depth_img[(depth_img < loc.depth_range[0]) | (depth_img > loc.depth_range[1])] = 0.0
			# To tensor
			rgb_img_tensor = rgb_image_to_tensor(rgb_img, resize, normalized=False)
			depth_img_tensor = depth_image_to_tensor(depth_img, depth_scale=1.0)
			# Intrinsic matrix
			raw_K = np.array(camera_info_msg.K).reshape((3, 3))
			raw_img_size = (int(camera_info_msg.width), int(camera_info_msg.height))
			if resize is not None:
				K = correct_intrinsic_scale(raw_K, resize[0] / raw_img_size[0], resize[1] / raw_img_size[1])
				img_size = (int(resize[0]), int(resize[1]))
			else:
				K = raw_K
				img_size = raw_img_size

			# Create observation node
			with torch.no_grad():
				desc = to_numpy(loc.vpr_model(rgb_img_tensor.unsqueeze(0).to(args.device)))
			obs_node = ImageNode(obs_id, rgb_img_tensor, depth_img_tensor, desc,
								 rgb_img_time, np.zeros(3), np.array([0, 0, 0, 1]),
								 K, img_size, None, None)
			obs_node.set_raw_intrinsics(raw_K, raw_img_size)

			loc.curr_obs_node = obs_node

			"""Perform global localization via. visual place recognition"""
			if not loc.has_global_pos:
				loc_start_time = time.time()
				result = loc.perform_global_loc(save_viz=False)
				rospy.loginfo(f"Global localization cost: {time.time() - loc_start_time:.3f}s")
				if result['succ']:
					matched_map_id = result['map_id']
					loc.has_global_pos = True
					loc.ref_map_node = loc.image_graph.get_node(matched_map_id)
					loc.curr_obs_node.set_pose(loc.ref_map_node.trans, loc.ref_map_node.quat)
					rospy.logwarn(f'Found VPR Node in global position: {matched_map_id}')
				else:
					rospy.logwarn('Failed to determine the global position since no VPR results.')
			else:
				# Initialize the current transformation using the historical fused poses
				idx_closest, stamped_pose_closest = fused_poses.find_closest(loc.curr_obs_node.time)
				# No fused poses available
				if idx_closest is None:
					init_trans, init_quat = loc.ref_map_node.trans, loc.ref_map_node.quat
				# Use the closest fused pose as the initial guess
				else:
					init_trans, init_quat = convert_matrix_to_vec(stamped_pose_closest[1])
				
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

			"""Perform local localization via. image matching"""
			if loc.has_global_pos:
				loc_start_time = time.time()
				result = loc.perform_local_loc()
				rospy.loginfo(f"Local localization cost: {time.time() - loc_start_time:.3f}s")
				if result['succ']:
					T_w_obs = result['T_w_obs']
					trans, quat = convert_matrix_to_vec(T_w_obs, 'xyzw')
					loc.curr_obs_node.set_pose(trans, quat)
					loc.has_local_pos = True
					rospy.logwarn(f'Estimated Poses: {trans.T}\n')
				else:
					loc.has_local_pos = False
					rospy.logwarn('[Fail] to determine the local position.\n')
				
			loc.publish_message()
			r.sleep()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.map_path, 'tmp/output_ros_loc_pipeline'))

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, out_dir)
	loc_pipeline.init_vpr_model()
	loc_pipeline.init_img_matcher()
	loc_pipeline.init_pose_solver()
	loc_pipeline.read_covis_graph_from_files()

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