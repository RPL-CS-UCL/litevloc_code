#! /usr/bin/env python

"""
Usage: 
python ros_global_planner.py \
--dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
--image_size 288 512 --device cuda \
--vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 512 --save_descriptors \
--num_preds_to_save 3 
"""

import os
import pathlib
import numpy as np
import torch
import rospy
import cv2
import threading
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int16

from point_graph import PointGraphLoader as GraphLoader
from image_node import ImageNode
from point_node import PointNode
from sensor_msgs.msg import Image
from loc_pipeline import LocPipeline
from utils.utils_image import rgb_image_to_tensor
from utils.utils_pipeline import parse_arguments, setup_log_environment
from utils.utils_shortest_path import dijk_shortest_path

# TODO(gogojjh):
import pycpptools.src.python.utils_ros as pytool_ros

class GlobalPlanner:
	def __init__(self, args):
		self.args = args

		self.loc_pipeline = None
		self.img_lock = threading.Lock()

		# Initialize variables
		self.plan_start_node = None
		self.plan_goal_node = None
		self.subgoals = []
		self.robot_id = 0
		self.conv_dist = 0.5

		# Sensor data
		self.robot_node = None
		self.manual_goal_img = None
		self.is_goal_init = False
		self.frame_id_map = 'map'

		# Planner status
		self.planner_status = Int16() 
		self.planner_status.data = 0 # 0: No goal for starting; 1: Find a path; 2: Reach a subgoal; 3: Fail to plan a path, and then set to 0

	def initalize_ros(self):
		# ROS subscriber
		rospy.Subscriber('/goal_image', Image, self.goal_img_callback, queue_size=1)
		rospy.Subscriber('/pose_fusion/odometry', Odometry, self.odom_callback, queue_size=1)

		# ROS publisher
		self.pub_shortest_path = rospy.Publisher('/graph/shortest_path', MarkerArray, queue_size=1)
		self.pub_waypoint = rospy.Publisher('/vloc/way_point', PointStamped, queue_size=1)
		self.status_pub = rospy.Publisher('/global_planner/status', Int16, queue_size=10)

	def read_map_from_file(self):
		map_root = Path(self.args.map_path)
		self.point_graph = GraphLoader.load_data(map_root, edge_type='trav')
		self.map_node_position = np.array([node.trans for _, node in self.point_graph.nodes.items()])
		rospy.loginfo(f"Loaded {self.point_graph} from {data_path}")

	def publish_path(self):
		header = Header(stamp=rospy.Time.now(), frame_id=f'{self.frame_id_map}_graph')
		pytool_ros.ros_vis.publish_shortest_path(self.subgoals, header, self.pub_shortest_path)

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
		self.status_pub.publish(self.planner_status)

	def perform_planning(self, robot_node):
		resize = self.args.image_size
		# Finding the goal node via. visual place recognition"""
		if not self.is_goal_init:
			self.planner_status.data = 0
			if self.manual_goal_img is not None:
				# process goal image
				self.img_lock.acquire()
				img_tensor = rgb_image_to_tensor(self.manual_goal_img, resize, normalized=False)
				self.img_lock.release()
				with torch.no_grad():
					desc = self.loc_pipeline.vpr_model(img_tensor.unsqueeze(0).to(self.args.device)).cpu().numpy()
				obs_node = ImageNode(0, img_tensor, None, desc, 
									rospy.Time.now().to_sec(), np.zeros(3), np.array([0, 0, 0, 1]), 
									None, resize, None, None)
				self.loc_pipeline.curr_obs_node = obs_node

				# find the goal image
				result = self.loc_pipeline.perform_global_loc(save=False)
				self.manual_goal_img = None
				self.loc_pipeline.curr_obs_node = None
				if not result['succ']:
					rospy.loginfo('No goal node found, need to wait other goal image')
					return

				goal_node = self.point_graph.get_node(result['map_id'])
				rospy.loginfo(f'Found goal node: {goal_node.id}')
				map_id = np.argmin(np.linalg.norm(self.map_node_position - robot_node.trans, axis=1))
				start_node = self.point_graph.get_node(map_id)

				# shortest path planning
				tra_distance, tra_path = \
					dijk_shortest_path(self.point_graph, start_node, goal_node)
				if tra_distance != float('inf'):
					for i in range(len(tra_path) - 1):
						node = tra_path[i]
						node_next = tra_path[i + 1]
						node.add_next_node(node_next)
					self.subgoals = []
					for node in tra_path:
						self.subgoals.append(node)
					out_str = 'Success to plan a path'
					out_str += f'Travel distance of the shortest path: {tra_distance:.3f}m\n'
					out_str += f'Shortest path: ' + ' -> '.join([str(node.id) for node in self.subgoals])
					rospy.logwarn(out_str)
					self.is_goal_init = True
					self.planner_status.data = 1
				else:
					rospy.logwarn('Fail to plan a path')
					self.planner_status.data = 3

		# Perform shortest path planning
		if self.is_goal_init:
			subgoal = self.subgoals[0]
			# Check whetherthe subgoal less than converage range
			dis_goal = np.linalg.norm(robot_node.trans - subgoal.trans)
			if dis_goal < self.conv_dist:
				self.subgoals.pop(0) # Remove the first subgoal
				rospy.loginfo(f'[Global Planning] SubGoal {subgoal.id} arrived')
			# Reach the goal
			if len(self.subgoals) == 0:
				self.is_goal_init = False
				# planner status -> Success
				self.planner_status.data = 2
				return
			# Publish the closest subgoal as waypoint
			self.publish_path()
			self.publish_waypoint(self.subgoals[0])

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
	global_planner.frame_id_map = 'vloc_map'
	global_planner.main_freq = 1.0
	global_planner.conv_dist = 3.0

	img_idx = 0
	map_root = Path(args.map_path)
	while not rospy.is_shutdown():
		path_goal_img = str(map_root / 'goal_images' / f'goal_img_{img_idx}.jpg')
		print(f'Loading goal image: {path_goal_img}')
		if os.path.exists(path_goal_img):
			global_planner.is_goal_init = False
			global_planner.manual_goal_img = cv2.imread(path_goal_img)

			curr_position = global_planner.map_node_position[0]
			odom_msg = Odometry()
			odom_msg.header.stamp = rospy.Time.now()
			odom_msg.pose.pose.position.x = curr_position[0]
			odom_msg.pose.pose.position.y = curr_position[1]
			odom_msg.pose.pose.position.z = curr_position[2]
			odom_msg.pose.pose.orientation.x = 0
			odom_msg.pose.pose.orientation.y = 0
			odom_msg.pose.pose.orientation.z = 0
			odom_msg.pose.pose.orientation.w = 1
			time = odom_msg.header.stamp.to_sec()
			# TODO(gogojjh):
			trans, quat = pytool_ros.ros_msg.convert_rosodom_to_vec(odom_msg)
			robot_node = PointNode(global_planner.robot_id, None, time, trans, quat, None, None)
			global_planner.robot_id += 1
			global_planner.perform_planning(robot_node)

			img_idx += 1
		else:
			img_idx = 0
		rospy.Rate(1).sleep()
