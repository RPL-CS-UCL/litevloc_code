#! /usr/bin/env python
"""
Usage: 
python ros_vis_map.py \
--dataset_path /Rocket_ssd/dataset/data_litevloc/ucl_campus/vloc_ops_msg/out_map
"""

import numpy as np
import argparse
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray
import tf2_ros

from point_graph import PointGraphLoader as GraphLoader

import pycpptools.src.python.utils_ros as pytool_ros

class ROSVisMap:
	def __init__(self, args):
		self.args = args
		self.frame_id_map = 'vloc_map'

	def initalize_ros(self):
		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		self.br = tf2_ros.TransformBroadcaster()

	def read_map_from_file(self):
		data_path = self.args.dataset_path
		self.point_graph = GraphLoader.load_data(data_path)
		print(f"Loaded {self.point_graph} from {data_path}")

	def perform_visualization(self):
		while not rospy.is_shutdown():
			header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id_map)
			tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, f"{self.frame_id_map}_graph")
			self.br.sendTransform(tf_msg)
			header = Header(stamp=header.stamp, frame_id=f"{self.frame_id_map}_graph")
			pytool_ros.ros_vis.publish_graph(self.point_graph, header, self.pub_graph, self.pub_graph_poses)
			rospy.Rate(1).sleep()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Visualize the map in ROS",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	args, unknown = parser.parse_known_args()

	rospy.init_node('ros_vis_map', anonymous=True)
	
	# Initialize the ros visualization map
	ros_vis_map = ROSVisMap(args)
	ros_vis_map.read_map_from_file()
	ros_vis_map.initalize_ros()
	ros_vis_map.perform_visualization()