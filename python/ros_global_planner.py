#! /opt/conda/envs/topo_loc/bin/python

"""
Usage: 
python ros_global_planner.py \
--dataset_path /Rocket_ssd/dataset/data_topo_loc/matterport3d/vloc_17DRP5sb8fy/out_map \
--image_size 288 512 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors \
--num_preds_to_save 3 
"""

import os
import pathlib
import logging
import numpy as np
import torch
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

from point_graph import PointGraphLoader as GraphLoader
from image_node import ImageNode
from point_node import PointNode
from sensor_msgs.msg import Image
from loc_pipeline import LocPipeline
from utils.utils_image import rgb_image_to_tensor
from utils.utils_pipeline import parse_arguments, setup_log_environment

import pycpptools.src.python.utils_algorithm as pytool_alg
import pycpptools.src.python.utils_ros as pytool_ros
import threading

class GlobalPlanner:
	def __init__(self, args):
		self.args = args

		self.loc_pipeline = None
		self.img_lock = threading.Lock()

		# Initialize variables
		self.plan_start_node = None
		self.plan_goal_node = None
		self.planner_path = []
		self.robot_id = 0
		self.conv_dist = 0.5

		# Sensor data
		self.robot_node = None
		self.manual_goal_img = None
		self.is_goal_init = False
		self.frame_id_map = 'map'

	def initalize_ros(self):
		# ROS subscriber
		rospy.Subscriber('/goal_image', Image, self.goal_img_callback, queue_size=1)
		rospy.Subscriber('/pose_fusion/odometry', Odometry, self.odom_callback, queue_size=1)

		# ROS publisher
		self.pub_shortest_path = rospy.Publisher('/graph/shortest_path', MarkerArray, queue_size=1)
		self.pub_waypoint = rospy.Publisher('/way_point', PointStamped, queue_size=1)

	def read_map_from_file(self):
		data_path = self.args.dataset_path
		self.point_graph = GraphLoader.load_data(data_path)
		self.map_node_poses = np.array([node.trans for _, node in self.point_graph.nodes.items()])
		logging.info(f"Loaded {self.point_graph} from {data_path}")

	def publish_path(self):
		header = Header(stamp=rospy.Time.now(), frame_id=f'{self.frame_id_map}_graph')
		pytool_ros.ros_vis.publish_shortest_path(self.planner_path, header, self.pub_shortest_path)

	def publish_waypoint(self, subgoal_node):
		header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id_map)
		waypoint_pos = subgoal_node.trans
		pytool_ros.ros_vis.publish_waypoint(waypoint_pos, header, self.pub_waypoint)

	def goal_img_callback(self, img_msg):
		if self.manual_goal_img is None:
			self.img_lock.acquire()
			rospy.loginfo('Receive goal image')
			self.manual_goal_img = pytool_ros.ros_msg.convert_rosimg_to_cvimg(img_msg)
			self.img_lock.release()

	def odom_callback(self, odom_msg):
		time = odom_msg.header.stamp.to_sec()
		trans, quat = pytool_ros.ros_msg.convert_rosodom_to_vec(odom_msg)
		robot_node = PointNode(self.robot_id, None, time, trans, quat, None, None)
		self.robot_id += 1
		self.perform_planning(robot_node)

	def perform_planning(self, robot_node):
		resize = self.args.image_size
		# Finding the goal node via. visual place recognition"""
		if not self.is_goal_init:
			if self.manual_goal_img is not None:
				self.img_lock.acquire()
				img_tensor = rgb_image_to_tensor(self.manual_goal_img, resize, normalized=False)
				self.img_lock.release()
				with torch.no_grad():
					desc = self.loc_pipeline.vpr_model(img_tensor.unsqueeze(0).to(args.device)).cpu().numpy()
				obs_node = ImageNode(0, img_tensor, None, desc, 
									rospy.Time.now().to_sec(), np.zeros(3), np.array([0, 0, 0, 1]), 
									None, resize, None, None)
				self.loc_pipeline.curr_obs_node = obs_node
				result = self.loc_pipeline.perform_global_loc(save=False)
				if result['succ']:
					self.is_goal_init = True
					self.plan_goal_node = self.point_graph.get_node(result['map_id'])
					rospy.loginfo(f'Found goal node: {self.plan_goal_node.id}')
				else:
					rospy.loginfo('No goal node found, need to wait other goal image')
				self.manual_goal_img = None
				self.loc_pipeline.curr_obs_node = None

		# Perform shortest path planning
		if self.is_goal_init:
			# check goal less than converage range
			dis_goal = np.linalg.norm(robot_node.trans - self.plan_goal_node.trans)
			if dis_goal < self.conv_dist:
				self.is_goal_init = False
				self.plan_goal_node = None
				rospy.loginfo('[Global Planning] Goal arrived')
				return True

			# shortest path planning
			map_id = np.argmin(np.linalg.norm(self.map_node_poses - robot_node.trans, axis=1))
			self.plan_start_node = self.point_graph.get_node(map_id)
			tra_distance, tra_path = \
				pytool_alg.sp.dijk_shortest_path(self.point_graph, self.plan_start_node, self.plan_goal_node)
			if tra_distance != float('inf'):
				self.planner_path = tra_path
				for i in range(len(tra_path) - 1):
					node = tra_path[i]
					node_next = tra_path[i + 1]
					node.add_next_node(node_next)
				out_str =  f"Travel distance of the shortest path: {tra_distance:.3f}m\n"
				out_str += f"Shortest path: " + " -> ".join([str(node.id) for node in tra_path])
				rospy.loginfo(out_str)
				self.publish_path()
				if self.plan_start_node.id == self.plan_goal_node.id:
					self.publish_waypoint(self.plan_goal_node)
				else:
					self.publish_waypoint(tra_path[1])
		rospy.Rate(self.main_freq).sleep()

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_global_planner'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the global planner
	global_planner = GlobalPlanner(args)
	global_planner.read_map_from_file()

	# Initialize the localization pipeline
	global_planner.loc_pipeline = LocPipeline(args, log_dir)
	global_planner.loc_pipeline.init_vpr_model()
	global_planner.loc_pipeline.read_map_from_file()
	
	rospy.init_node('global_planner', anonymous=True)
	global_planner.loc_pipeline.initalize_ros()
	global_planner.initalize_ros()
	global_planner.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	global_planner.main_freq = rospy.get_param('~main_freq', 1)
	global_planner.conv_dist = rospy.get_param('~conv_dist', 0.5)

	rospy.spin()
