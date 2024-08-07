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
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros
import message_filters

from matching.utils import to_numpy
from utils.utils_vpr_method import initialize_vpr_model, perform_knn_search
from utils.utils_vpr_method import save_visualization as save_vpr_visualization
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image import rgb_image_to_tensor, depth_image_to_tensor
from utils.utils_pipeline import *
from utils.pose_solver import get_solver
from utils.pose_solver_default import cfg
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

from loc_pipeline import LocPipeline
import queue
import threading
import PIL

if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

rgb_depth_queue = queue.Queue()

def rgb_depth_image_callback(rgb_img_msg, depth_img_msg, camera_info_msg):
    rgb_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(rgb_img_msg)
    depth_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(depth_img_msg)
    if depth_img_msg.encoding == "mono16": depth_img *= 0.001
    K = np.array(camera_info_msg.K).reshape((3, 3))
    img_size = (camera_info_msg.width, camera_info_msg.height)
    rgb_depth_queue.put((rgb_img, depth_img, K, img_size))

def perform_localization(loc: LocPipeline, args):
    obs_id = 0
    while not rospy.is_shutdown():
        if not rgb_depth_queue.empty():
            img_size = args.image_size
            rgb_img, depth_img, raw_K, raw_img_size = rgb_depth_queue.get()
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
            vpr_start_time = time.time()
            with torch.no_grad():
                desc = loc.vpr_model(rgb_img_tensor.unsqueeze(0).to(args.device)).cpu().numpy()
            print(f"Extract VPR descriptors cost: {time.time() - vpr_start_time:.3f}s")
            
            obs_node = ImageNode(obs_id, rgb_img_tensor, depth_img_tensor, desc,
                                 0, np.zeros(3), np.array([0, 0, 0, 1]),
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
                init_trans, init_quat = loc.last_obs_node.trans, loc.last_obs_node.quat
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
    loc_pipeline.read_map_from_file()

    rospy.init_node('ros_loc_pipeline_simu', anonymous=True)
    loc_pipeline.setup_ros_publishers()

    # Subscribe to RGB, depth images, and odometry
    rgb_sub = message_filters.Subscriber('/habitat_camera/color/image', Image)
    depth_sub = message_filters.Subscriber('/habitat_camera/depth/image', Image)
    camera_info_sub = message_filters.Subscriber('/habitat_camera/color/camera_info', CameraInfo)

    # TODO(gogojjh):
    # odom_sub = rospy.Subscriber('/prior_odometry', Odometry, odom_callback)

    # Synchronize the topics
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, camera_info_sub], queue_size=10, slop=0.1)
    ts.registerCallback(rgb_depth_image_callback)

    # Start the localization thread
    localization_thread = threading.Thread(target=perform_localization, args=(loc_pipeline, args, ))
    localization_thread.start()

    rospy.spin()