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
	--min_solver_inliers_thre 300
"""

# General
import os
import sys
import argparse
import pathlib
import numpy as np
import torch
import time
import queue
import threading
import traceback
from typing import Callable, Optional

# ROS
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_srvs.srv import Trigger, TriggerResponse
import message_filters

# Others
from utils.utils_image import rgb_image_to_tensor, depth_image_to_tensor, to_numpy
from utils.utils_pipeline import parse_arguments
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, compute_pose_error, correct_intrinsic_scale
from utils.utils_ros import ros_msg
from utils.utils_ros_image import decode_depth_msg, decode_rgb_msg, filter_depth_range
from utils.utils_stamped_poses import StampedPoses
from image_node import ImageNode
from loc_pipeline import LocPipeline

fused_poses = StampedPoses()
rgb_depth_queue = queue.Queue()
lock = threading.Lock()

# After this many consecutive local-localization failures, drop back to global
# localization instead of waiting forever for a fused-odometry-based reset.
MAX_LOCAL_LOC_FAILS = 3

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

def _build_force_global_loc_callback(loc: LocPipeline) -> Callable[[object], object]:
	"""Build the /vloc/force_global_loc Trigger service callback for this loc pipeline.

	The callback only flips an Event; localization state is owned by the
	localization thread, so process_frame applies the actual reset at the
	start of the next frame it processes.
	"""
	def _callback(_req):
		loc.force_global_event.set()
		return TriggerResponse(success=True, message="global localization will rerun on the next frame")
	return _callback

def process_frame(
	loc: LocPipeline,
	args: argparse.Namespace,
	rgb_img_msg,
	depth_img_msg,
	camera_info_msg,
	desc_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
	"""Process one synchronized (rgb, depth, camera_info) frame and publish the result."""
	# A forced relocalization request (service callback) is applied at the start of the
	# next frame, since localization state is only touched from this thread.
	if loc.force_global_event.is_set():
		loc.force_global_event.clear()
		loc.has_global_pos = False
		loc.ref_map_node = None
		loc.local_fail_count = 0
		loc.curr_query_descs = []
		rospy.logwarn('Forced global relocalization requested. Resetting global position.')

	# Reset every frame: only this frame's successful local localization may set it True.
	# Otherwise a stale True (from before a reset to global) would publish a (0,0,0) pose.
	loc.has_local_pos = False

	resize = args.image_size # WxH
	loc.child_frame_id = rgb_img_msg.header.frame_id
	rgb_img_time = rgb_img_msg.header.stamp.to_sec()

	rgb_img = decode_rgb_msg(rgb_img_msg)
	depth_img = decode_depth_msg(depth_img_msg)
	depth_img = filter_depth_range(depth_img, *loc.depth_range)

	# To tensor
	rgb_img_tensor = rgb_image_to_tensor(rgb_img, resize, normalized=False)
	depth_img_tensor = depth_image_to_tensor(depth_img, depth_scale=1.0)

	# Intrinsic matrix
	raw_K = np.array(camera_info_msg.K).reshape((3, 3))
	raw_img_size = (int(camera_info_msg.width), int(camera_info_msg.height))
	if resize is not None:
		K = correct_intrinsic_scale(
			raw_K, resize[0] / raw_img_size[0], resize[1] / raw_img_size[1]
		)
		img_size = (int(resize[0]), int(resize[1]))
	else:
		K = raw_K
		img_size = raw_img_size

	# VPR descriptor. If desc_fn is given, it computes it directly from the
	# original-size RGB (consistent with the mapping side); otherwise fall back
	# to loc.vpr_model on the resized tensor (original upstream behavior).
	if desc_fn is not None:
		desc = np.asarray(desc_fn(rgb_img), dtype=np.float32).reshape(1, -1)
	else:
		with torch.no_grad():
			desc = to_numpy(loc.vpr_model(rgb_img_tensor.unsqueeze(0).to(args.device)))

	# Create observation node
	obs_node = ImageNode(
		loc.obs_id, rgb_img_tensor, depth_img_tensor, desc,
		rgb_img_time, np.zeros(3), np.array([0, 0, 0, 1]),
		K, img_size,
		f"seq/{loc.obs_id:06d}.color.jpg",
		f"seq/{loc.obs_id:06d}.depth.png"
	)
	obs_node.set_raw_intrinsics(raw_K, raw_img_size)
	loc.curr_obs_node = obs_node
	loc.obs_id += 1

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
			loc.local_fail_count = 0
			rospy.logwarn(f'Estimated Poses: {trans.T}\n')
		else:
			loc.has_local_pos = False
			loc.local_fail_count += 1
			rospy.logwarn('[Fail] to determine the local position.\n')
			if loc.local_fail_count >= MAX_LOCAL_LOC_FAILS:
				rospy.logwarn(
					f'Local localization failed {loc.local_fail_count} times in a row. '
					'Resetting to global localization.'
				)
				loc.has_global_pos = False
				loc.ref_map_node = None
				loc.local_fail_count = 0

	loc.publish_message()

def perform_localization(
	loc: LocPipeline,
	args: argparse.Namespace,
	desc_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
	r = rospy.Rate(loc.main_freq)
	while not rospy.is_shutdown():
		if not rgb_depth_queue.empty():
			"""Get the latest RGB, depth images, and camera info"""
			lock.acquire()
			rgb_img_msg, depth_img_msg, camera_info_msg = rgb_depth_queue.get()
			lock.release()
			try:
				process_frame(loc, args, rgb_img_msg, depth_img_msg, camera_info_msg, desc_fn)
			except Exception:
				rospy.logerr(f"Error while processing frame:\n{traceback.format_exc()}")
		# Always sleep, even when the queue is empty, so an idle loop does not spin a CPU core.
		r.sleep()

def main(
	make_desc_fn: Optional[Callable[[argparse.Namespace], Callable[[np.ndarray], np.ndarray]]] = None,
) -> None:
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.map_path, 'tmp/output_ros_loc_pipeline'))
	config = dict(
		resize=args.image_size, depth_scale=args.depth_scale,
		load_rgb=True, load_depth=False, normalized=False,
	)

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, out_dir)
	if make_desc_fn is not None:
		# A caller-supplied descriptor hook replaces loc.vpr_model entirely, so skip
		# loading the upstream VPR model (avoids loading the same model twice on GPU).
		desc_fn = make_desc_fn(args)
	else:
		loc_pipeline.init_vpr_model()
		desc_fn = None
	loc_pipeline.init_img_matcher()
	loc_pipeline.init_pose_solver()
	loc_pipeline.read_covis_graph_from_files(config)
	loc_pipeline.init_vpr_match_model()

	# Per-run state used by process_frame
	loc_pipeline.obs_id = 0
	loc_pipeline.local_fail_count = 0
	loc_pipeline.force_global_event = threading.Event()

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

	# Service to force a global relocalization on the next processed frame.
	force_global_loc_srv = rospy.Service(
		'/vloc/force_global_loc', Trigger, _build_force_global_loc_callback(loc_pipeline)
	)

	# Start the localization thread
	localization_thread = threading.Thread(target=perform_localization, args=(loc_pipeline, args, desc_fn))
	localization_thread.start()

	rospy.spin()

if __name__ == '__main__':
	main()
