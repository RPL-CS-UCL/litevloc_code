#! /usr/bin/env python

import os
import numpy as np
import argparse

import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray
import tf2_ros

from point_graph import PointGraphLoader as GraphLoader
import pycpptools.src.python.utils_ros as pytool_ros

class ROSPublishGraph:
	def __init__(self, args):
		self.args = args

	def initialize_ros(self):
		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		self.br = tf2_ros.TransformBroadcaster()

	def read_map_from_file(self):
		self.point_graph = GraphLoader.load_data(self.args.map_path)
		print('Loaded point graph from {}'.format(self.args.map_path))
		print('Number of nodes: {}'.format(len(self.point_graph.nodes)))

	def publish_message(self):
		header = Header(stamp=rospy.Time.now(), frame_id='map')
		tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(np.array([0, 2.0, 0.0]), np.array([0, 0, 0, 1]), header, 'map_graph')
		self.br.sendTransform(tf_msg)

		header = Header(stamp=rospy.Time.now(), frame_id='map_graph')
		pytool_ros.ros_vis.publish_graph(self.point_graph, header, self.pub_graph, self.pub_graph_poses)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="ROSPublishGraph",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument("--map_path", type=str, default="matterport3d", help="path to map_path")
	args, unknown = parser.parse_known_args()

	ros_publish_graph = ROSPublishGraph(args)
	ros_publish_graph.read_map_from_file()

	rospy.init_node('ros_publish_graph', anonymous=True)
	ros_publish_graph.initialize_ros()

	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		ros_publish_graph.publish_message()
		rate.sleep()
