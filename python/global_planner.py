#! /usr/bin/env python

"""
Usage: 
python python/global_planner.py \
	--map_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--image_size 512 288 --device cuda \
	--vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 512
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
from utils.utils_image import rgb_image_to_tensor, to_numpy
from utils.utils_pipeline import parse_arguments
from utils.utils_shortest_path import dijk_shortest_path
from utils.utils_ros import ros_msg, ros_vis

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

	def read_trav_graph_from_files(self):
		map_root = pathlib.Path(self.args.map_path)
		self.point_graph = GraphLoader.load_data(map_root, edge_type='trav')
		self.map_node_position = np.array([node.trans for _, node in self.point_graph.nodes.items()])
		rospy.loginfo(str(self.point_graph))

	def publish_path(self, subgoals, timestamp):
		header = Header(stamp=timestamp, frame_id=f'{self.frame_id_map}_graph')
		ros_vis.publish_shortest_path(subgoals, header, self.pub_shortest_path)

	def publish_waypoint(self, subgoal_node, timestamp):
		header = Header(stamp=timestamp, frame_id=self.frame_id_map)
		waypoint_pos = subgoal_node.trans
		ros_vis.publish_waypoint(waypoint_pos, header, self.pub_waypoint)

	def goal_img_callback(self, img_msg):
		if self.manual_goal_img is None:
			self.img_lock.acquire()
			rospy.loginfo('Receive goal image')
			self.manual_goal_img = ros_msg.convert_rosimg_to_cvimg(img_msg)
			self.img_lock.release()

	def odom_callback(self, odom_msg):
		time = odom_msg.header.stamp.to_sec()
		trans, quat = ros_msg.convert_rosodom_to_vec(odom_msg)
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
					desc = to_numpy(self.loc_pipeline.vpr_model(img_tensor.unsqueeze(0).to(self.args.device)))

				obs_node = ImageNode(
					0, img_tensor, None, desc, 
					rospy.Time.now().to_sec(), 
					np.zeros(3), np.array([0, 0, 0, 1]), 
					None, resize, None, None
				)
				self.loc_pipeline.curr_obs_node = obs_node

				# find the goal image
				result = self.loc_pipeline.perform_global_loc(save_viz=False)
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
			viz_subgoals = self.subgoals.copy()

			subgoal = self.subgoals[0]
			# Check whetherthe subgoal less than converage range
			dis_goal, _ = robot_node.compute_distance(subgoal)
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
			timestamp = rospy.Time.now()
			self.publish_path(viz_subgoals, timestamp)
			self.publish_waypoint(self.subgoals[0], timestamp)

		rospy.Rate(self.main_freq).sleep()

if __name__ == '__main__':
	args = parse_arguments()

	# Set map path and log folder
	map_root = pathlib.Path(args.map_path)
	
	# Initialize the global planner
	global_planner = GlobalPlanner(args)
	global_planner.read_trav_graph_from_files()

	# Initialize the localization pipeline
	global_planner.loc_pipeline = LocPipeline(args, map_root/'output_global_planner')
	global_planner.loc_pipeline.init_vpr_model()
	global_planner.loc_pipeline.read_covis_graph_from_files()
	
	rospy.init_node('global_planner', anonymous=True)
	global_planner.loc_pipeline.initalize_ros()
	global_planner.initalize_ros()
	global_planner.frame_id_map = 'vloc_map'
	global_planner.main_freq = 1.0
	global_planner.conv_dist = 3.0

	# Initialize the ROS interface
	from geometry_msgs.msg import PoseArray
	import tf2_ros

	br = tf2_ros.TransformBroadcaster()
	pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
	pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)

	img_idx = 0
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
			trans, quat = ros_msg.convert_rosodom_to_vec(odom_msg)
			robot_node = PointNode(global_planner.robot_id, time, trans, quat, None)
			global_planner.robot_id += 1
			global_planner.perform_planning(robot_node)

			# Publish other ROS message
			header = Header(stamp=odom_msg.header.stamp, frame_id=global_planner.frame_id_map)
			tf_msg = ros_msg.convert_vec_to_rostf(
				np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, f"{global_planner.frame_id_map}_graph"
			)
			br.sendTransform(tf_msg)
			header.frame_id += '_graph'
			ros_vis.publish_graph(
				global_planner.point_graph, header, pub_graph, pub_graph_poses
			)

			img_idx += 1
		else:
			img_idx = 0

		rospy.Rate(1).sleep()
