#! /usr/bin/env python

"""
Usage: 
python python/global_planner.py \
	--map_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--image_size 512 288 --device cuda \
	--vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 256 \
	--goal_image goal_image.jpg (optional)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
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
from std_msgs.msg import String
from sensor_msgs.msg import Image

from point_graph import PointGraphLoader as GraphLoader
from image_node import ImageNode
from point_node import PointNode
from loc_pipeline import LocPipeline
from utils.utils_image import rgb_image_to_tensor, to_numpy
from utils.utils_pipeline import parse_arguments
from utils.utils_ros import ros_msg, ros_vis
from litevloc_topo.goals import KeyframeGoal, ObjectGoal
from litevloc_topo.graph import Pose2D, TopoGraphError, load_topo_graph
from litevloc_topo.planner import TopoPlanner

# Go2 相机在机身中心前方的距离(米),与 nav_mapping.map_route 一致
CAMERA_FORWARD_M = 0.1582

class GlobalPlanner:
	def __init__(self, args):
		self.args = args

		self.loc_pipeline = None
		self.img_lock = threading.Lock()
		self.object_lock = threading.Lock()

		# Initialize variables
		self.plan_start_node = None
		self.plan_goal_node = None
		self.subgoals = []
		self.robot_id = 0
		self.conv_dist = 0.5

		# Sensor data
		self.robot_node = None
		self.manual_goal_img = None
		self.manual_goal_object = None  # /goal_object 收到的 objects.json 实例 id
		self.is_goal_init = False
		self.frame_id_map = 'map'

		# Planner status
		self.planner_status = Int16() 
		self.planner_status.data = 0 # 0: No goal for starting; 1: Find a path; 2: Reach a subgoal; 3: Fail to plan a path, and then set to 0

	def initalize_ros(self):
		# ROS subscriber
		rospy.Subscriber('/goal_image', Image, self.goal_img_callback, queue_size=1)
		rospy.Subscriber('/goal_object', String, self.goal_object_callback, queue_size=1)
		rospy.Subscriber('/pose_fusion/odometry', Odometry, self.odom_callback, queue_size=1)

		# ROS publisher
		self.pub_shortest_path = rospy.Publisher('/graph/shortest_path', MarkerArray, queue_size=1)
		self.pub_waypoint = rospy.Publisher('/vloc/way_point', PointStamped, queue_size=1)
		self.status_pub = rospy.Publisher('/global_planner/status', Int16, queue_size=10)

	def read_trav_graph_from_files(self):
		map_root = pathlib.Path(self.args.map_path)
		self.point_graph = GraphLoader.load_data(map_root, edge_type='trav')
		self.map_node_position = np.array([node.trans for _, node in self.point_graph.nodes.items()])
		logging.info(f"Loading Traversiblity Graph: {str(self.point_graph)}")

		# litevloc_topo 规划专用图:只要平面(x, y, yaw),Dijkstra / 子目标推进都在这张图上做,
		# 和上面的 point_graph(给可视化、VPR 最近邻用)是两张独立的图,关键帧编号(行号)一致。
		self.topo_graph = load_topo_graph(map_root, camera_forward_m=CAMERA_FORWARD_M, edge_type='trav')
		self.planner = TopoPlanner(self.topo_graph, map_dir=map_root)

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

	def goal_object_callback(self, string_msg):
		# string_msg.data 是地图 objects.json 里某个物体的实例 id(比如 obj_3),直接当目标用,
		# 不用再走 VPR 图像匹配那一套。
		if self.manual_goal_object is None:
			self.object_lock.acquire()
			rospy.loginfo(f'Receive goal object: {string_msg.data}')
			self.manual_goal_object = string_msg.data
			self.object_lock.release()

	def odom_callback(self, odom_msg):
		time = odom_msg.header.stamp.to_sec()
		trans, quat = ros_msg.convert_rosodom_to_vec(odom_msg)
		robot_node = PointNode(self.robot_id, time, trans, quat, None)
		self.robot_id += 1

		self.perform_planning(robot_node)
		self.status_pub.publish(self.planner_status)

	def perform_planning(self, robot_node):
		resize = self.args.image_size
		# Finding the goal node via. visual place recognition, or straight from /goal_object"""
		if not self.is_goal_init:
			self.planner_status.data = 0
			goal = None

			if self.manual_goal_object is not None:
				# 物体目标不用 VPR,直接按实例 id 去 objects.json 查关键帧(litevloc_topo.goals.ObjectGoal)
				self.object_lock.acquire()
				instance_id = self.manual_goal_object
				self.manual_goal_object = None
				self.object_lock.release()
				goal = ObjectGoal(instance_id)

			elif self.manual_goal_img is not None:
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
					print('No goal node found, need to wait other goal image')
					return

				# result['map_id'] 就是 poses.txt 的行号,和 litevloc_topo 的关键帧编号同一套编号
				goal = KeyframeGoal(result['map_id'])

			if goal is not None:
				# shortest path planning:起点 = 离机身当前位置最近的关键帧,规划都在 litevloc_topo 里做
				start_pose = Pose2D(float(robot_node.trans[0]), float(robot_node.trans[1]), 0.0)
				try:
					route = self.planner.plan(start_pose, goal)
				except TopoGraphError as exc:
					print(f'Fail to plan a path: {exc}')
					self.planner_status.data = 3
				else:
					print(f'Found goal node: {route.kf_path[-1]}')
					# 复用现成的 RViz 可视化(ros_vis.publish_shortest_path/publish_waypoint 认 Node 对象),
					# 按关键帧编号(与 point_graph 的节点 id 同一套编号)从 point_graph 取回 Node、串好 next_node 链。
					tra_path = [self.point_graph.get_node(kf) for kf in route.kf_path]
					for i in range(len(tra_path) - 1):
						tra_path[i].add_next_node(tra_path[i + 1])
					self.subgoals = tra_path
					out_str = 'Success to plan a path\n'
					out_str += f'Travel distance of the shortest path: {route.length_m:.3f}m\n'
					out_str += 'Shortest path: ' + ' -> '.join([str(node.id) for node in self.subgoals])
					print(out_str)
					self.is_goal_init = True
					self.planner_status.data = 1
					##### DEBUG(gogojjh): User specify the goal image, only True during Debug
					if self.args.goal_image and self.args.viz:
						print('Visualize graph')
						self.visualize_graph(route.kf_path, self.args.goal_image)
					#####

		# Iteratively publish subgoals
		if self.is_goal_init:
			viz_subgoals = self.subgoals.copy()

			# TopoPlanner.advance 按 conv_dist 推进子目标,内部状态和 self.subgoals 按关键帧编号对齐
			robot_pose = Pose2D(float(robot_node.trans[0]), float(robot_node.trans[1]), 0.0)
			subgoal_kf = self.planner.advance(robot_pose, self.conv_dist)
			while self.subgoals and self.subgoals[0].id != subgoal_kf:
				arrived = self.subgoals.pop(0) # Remove the first subgoal
				print(f'[Global Planning] SubGoal {arrived.id} arrived')

			# Reach the goal
			if subgoal_kf is None:
				self.subgoals = []
				self.is_goal_init = False
				# planner status -> Success
				self.planner_status.data = 2
				return

			# Publish the closest subgoal as waypoint
			timestamp = rospy.Time.now()
			self.publish_path(viz_subgoals, timestamp)
			self.publish_waypoint(self.subgoals[0], timestamp)

		rospy.Rate(self.main_freq).sleep()

	def visualize_graph(self, shortest_path, goal_image_path):
		import re
		from utils.utils_setting_color_font import setting_font, acquire_color_palette 
		import matplotlib.pyplot as plt
		import cv2
		
		PALLETE = acquire_color_palette()
		setting_font(fontsize=20, titlesize=20, legend_fontsize=20)
		
		fig = plt.figure(figsize=(8, 9.2))
		# Display goal image if provided
		base_name = os.path.splitext(os.path.basename(goal_image_path))[0]
		match = re.search(r'_(\d{8})$', base_name)
		date_str = match.group(1) if match else None

		ax1 = plt.subplot(2, 1, 1)
		img = cv2.imread(goal_image_path)
		img_resize = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
		ax1.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
		ax1.axis('off')
		if date_str:
			year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
			ax_title = f"Goal Image Captured on {year}/{month}/{day}"
			ax1.set_title(ax_title, fontsize=20)

		# Plot all nodes in black
		ax2 = plt.subplot(2, 1, 2)
		# all_x = [pose.x for pose in self.topo_graph.poses.values()]
		# all_y = [pose.y for pose in self.topo_graph.poses.values()]
		# ax2.scatter(all_x, all_y, c='black', s=15)
		plt.subplots_adjust(hspace=0.04)

		# Plot all edges in black(节点坐标从 litevloc_topo 的 TopoGraph 取,不再读 point_graph)
		for kf, pose in self.topo_graph.poses.items():
			for neighbor_kf, _weight in self.topo_graph.neighbors(kf):
				neighbor_pose = self.topo_graph.poses[neighbor_kf]
				ax2.plot([pose.x, neighbor_pose.x], [pose.y, neighbor_pose.y], color='k', linewidth=3.0)

		# Highlight shortest path in red(shortest_path 是关键帧编号序列,如 Route.kf_path)
		if shortest_path:
			path_x = [self.topo_graph.poses[kf].x for kf in shortest_path[::2]]
			path_y = [self.topo_graph.poses[kf].y for kf in shortest_path[::2]]
			ax2.plot(
				path_x, path_y, color=PALLETE[0], linewidth=8, label='Shortest Path'
			)
			ax2.scatter(path_x[0], path_y[0], c=PALLETE[5], s=500, marker='*', label='Start Point', zorder=101)
			ax2.scatter(path_x[-1], path_y[-1], c=PALLETE[5], s=500, marker='^', label='End Point', zorder=101)
		
		# ax2.set_xlabel('X [m]')
		# ax2.set_ylabel('Y [m]')
		ax2.tick_params(axis='x', labelsize=18)
		ax2.tick_params(axis='y', labelsize=18)
		# ax2.invert_xaxis()
		# ax2.invert_yaxis()
		# ax2.set_xlim(min(start[0], end[0])-3, max(start[0], end[0])+3)
		# ax2.set_ylim(min(start[1], end[1])-3, max(start[1], end[1])+3)
		# ax2.set_title('Shortest Path on the Traversability Graph')
		ax2.grid(True, alpha=0.7, linestyle='--')
		# ax2.axis('equal')
		ax2.legend(fontsize=20, loc='lower left')
		
		# Save or show the plot
		map_name = os.path.splitext(os.path.basename(pathlib.Path(self.args.map_path)))[0]
		save_dir = pathlib.Path(goal_image_path).parent
		(save_dir / 'preds_shortestpath').mkdir(parents=True, exist_ok=True)
		goal_name = os.path.splitext(os.path.basename(goal_image_path))[0]
		save_path = str(save_dir / 'preds_shortestpath' / f'sp_{map_name}_{goal_name}.pdf')
		plt.savefig(save_path, dpi=300)
		save_path = save_path.replace('.pdf', '.png')
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f'Saved visualization to {save_path}')
		plt.close()


if __name__ == '__main__':
	args = parse_arguments()
	
	logging_level = logging.INFO
	logging.basicConfig(
		level=logging_level,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[logging.StreamHandler()]
	)

	config = dict(
		resize=args.image_size, depth_scale=args.depth_scale, 
		load_rgb=False, load_depth=False, normalized=False
	)

	# Set map path and log folder
	map_root = pathlib.Path(args.map_path)
	
	# Initialize the global planner
	global_planner = GlobalPlanner(args)
	global_planner.read_trav_graph_from_files()

	# Initialize the localization pipeline
	global_planner.loc_pipeline = LocPipeline(args, map_root/'tmp/output_global_planner')
	global_planner.loc_pipeline.init_vpr_model()
	global_planner.loc_pipeline.read_covis_graph_from_files(config)
	global_planner.loc_pipeline.init_vpr_match_model()
	
	rospy.init_node('global_planner', anonymous=True)
	global_planner.initalize_ros()
	global_planner.frame_id_map = 'vloc_map'
	global_planner.main_freq = 1.0
	global_planner.conv_dist = 3.0

	if args.goal_image:
		if os.path.exists(args.goal_image):
			global_planner.is_goal_init = False
			global_planner.manual_goal_img = cv2.imread(args.goal_image)
		else:
			exit()
	else:
		img_idx = 0 # User-identify
		path_goal_img = str(map_root / 'goal_images' / f'goal_img_{img_idx}.jpg')
		print(f'Loading goal image: {path_goal_img}')
		if os.path.exists(path_goal_img):
			global_planner.is_goal_init = False
			global_planner.manual_goal_img = cv2.imread(path_goal_img)
		else:
			raise ValueError(f'Not Exist Goal Image {path_goal_img}')

	trans, quat = global_planner.map_node_position[0], np.array([0, 0, 0, 1])
	robot_node = PointNode(global_planner.robot_id, rospy.Time.now().to_sec(), trans, quat, None)
	global_planner.perform_planning(robot_node)
	if args.viz: exit()

	# Publish other ROS message
	print('Publishing ROS Message for Visualization ...')
	from geometry_msgs.msg import PoseArray
	import tf2_ros

	br = tf2_ros.TransformBroadcaster()
	pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
	pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
	while not rospy.is_shutdown():
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

		header = Header(stamp=odom_msg.header.stamp, frame_id=global_planner.frame_id_map)
		tf_msg = ros_msg.convert_vec_to_rostf(
			np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, f"{global_planner.frame_id_map}_graph"
		)
		br.sendTransform(tf_msg)
		header.frame_id += '_graph'
		ros_vis.publish_graph(
			global_planner.point_graph, header, pub_graph, pub_graph_poses
		)
		rospy.Rate(1).sleep()
