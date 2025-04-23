#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseArray, PointStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
import tf2_ros
import tf.transformations as tf
import numpy as np

from . import ros_msg

def create_node_marker(node, header):
  marker = Marker()
  marker.header = header
  marker.ns = "nodes"
  marker.id = node.id
  marker.type = Marker.CUBE
  marker.action = Marker.ADD
  marker.pose.position.x = node.trans[0]
  marker.pose.position.y = node.trans[1]
  marker.pose.position.z = node.trans[2]
  marker.pose.orientation.x = node.quat[0]
  marker.pose.orientation.y = node.quat[0]
  marker.pose.orientation.z = node.quat[0]
  marker.pose.orientation.w = node.quat[0]
  marker.scale.x = 0.5
  marker.scale.y = 0.5
  marker.scale.z = 0.5
  marker.color.a = 0.7
  marker.color.r = 0.0
  marker.color.g = 1.0
  marker.color.b = 0.0
  return marker

def create_text_marker(text_id, position, text, header):
  marker = Marker()
  marker.header = header
  marker.ns = "text"
  marker.id = text_id
  marker.type = Marker.TEXT_VIEW_FACING
  marker.action = Marker.ADD
  marker.pose.position.x = position[0]
  marker.pose.position.y = position[1]
  marker.pose.position.z = position[2] + 0.5
  marker.scale.z = 0.5
  marker.color.a = 1.0
  marker.color.r = 1.0
  marker.color.g = 1.0
  marker.color.b = 1.0
  marker.text = text
  return marker

def create_edge_marker(node1, node2, edge_id, weight, header):
  marker = Marker()
  marker.header = header
  marker.ns = "edges"
  marker.id = edge_id
  marker.type = Marker.LINE_STRIP
  marker.action = Marker.ADD
  marker.scale.x = 0.03
  marker.color.a = 0.5
  marker.color.r = 0.0
  marker.color.g = 0.0
  marker.color.b = 1.0

  start_point = Point()
  start_point.x = node1.trans[0]
  start_point.y = node1.trans[1]
  start_point.z = node1.trans[2]

  end_point = Point()
  end_point.x = node2.trans[0]
  end_point.y = node2.trans[1]
  end_point.z = node2.trans[2]

  marker.points.append(start_point)
  marker.points.append(end_point)
  return marker

def publish_graph(graph, header, pub_graph, pub_graph_poses):
  # Publish graph node and edges
  marker_array = MarkerArray()
  for node_id, node in graph.nodes.items():
    node_marker = create_node_marker(node, header)
    marker_array.markers.append(node_marker)

    text_marker = create_text_marker(node.id, node.trans, f'{node_id}', header)
    marker_array.markers.append(text_marker)

  edge_id = 0
  for node in graph.nodes.values():
    for (to_node, weight) in node.edges.values():
      edge_marker = create_edge_marker(node, to_node, edge_id, weight, header)
      marker_array.markers.append(edge_marker)
      edge_id += 1

  pub_graph.publish(marker_array)

  # Publish graph poses
  poses = PoseArray()
  poses.header = header
  for node_id, node in graph.nodes.items():
    pose_stamped = ros_msg.convert_vec_to_rospose(node.trans, node.quat, header)
    poses.poses.append(pose_stamped.pose)
  pub_graph_poses.publish(poses)

def publish_shortest_path(path, header, pub_shortest_path):
  marker_array = MarkerArray()
  edge_id = 0
  for node in path:
    if node.get_next_node() is None: break
    edge_marker = create_edge_marker(node, node.get_next_node(), edge_id, 1.0, header)
    edge_marker.scale.x = 0.2
    edge_marker.color.a = 0.5
    edge_marker.color.r = 0.0
    edge_marker.color.g = 1.0
    edge_marker.color.b = 0.0
    marker_array.markers.append(edge_marker)
    edge_id += 1
  pub_shortest_path.publish(marker_array)

def publish_waypoint(waypoint_pos, header, pub_waypoint):
  point_stamped = PointStamped()
  point_stamped.header = header
  point_stamped.point.x = waypoint_pos[0]
  point_stamped.point.y = waypoint_pos[1]
  point_stamped.point.z = waypoint_pos[2]
  pub_waypoint.publish(point_stamped)

class TestRosVisualization:
  def __init__(self):
    pass

  def run_test(self):
    from utils.base_node import BaseNode as Node
    from utils.base_graph import BaseGraph as Graph

    rospy.init_node('test_ros_visualization', anonymous=True)

    pub_graph = rospy.Publisher('/topo_graph', MarkerArray, queue_size=10)
    pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
    pub_path = rospy.Publisher('/path', Path, queue_size=10)
    br = tf2_ros.TransformBroadcaster()    

    graph = Graph()
    graph.add_node(Node(0, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(1, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(2, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(3, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(4, np.random.rand(3, 1) * 5.0))
    graph.add_edge(graph.get_node(0), graph.get_node(1), 1)
    graph.add_edge(graph.get_node(0), graph.get_node(2), 2)
    graph.add_edge(graph.get_node(0), graph.get_node(3), 3)
    graph.add_edge(graph.get_node(3), graph.get_node(4), 5)
        
    trans = np.random.rand(3, 1) * 5.0
    quat = np.random.rand(4, 1)
    quat /= np.linalg.norm(quat)

    path_msg = Path()

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
      header = Header()
      header.stamp = rospy.Time.now()
      header.frame_id = "map"

      # Publish graph
      publish_graph(graph, pub_graph, header)

      # Publish odometry, path and tf messages
      child_frame_id = "camera"

      odom_msg = ros_msg.convert_vec_to_rosodom(trans, quat, header, child_frame_id)
      pub_odom.publish(odom_msg)
      
      pose_msg = ros_msg.convert_vec_to_rospose(trans, quat, header)
      path_msg.header = header
      path_msg.poses.append(pose_msg)
      pub_path.publish(path_msg)

      tf_msg = ros_msg.convert_vec_to_rostf(trans, quat, header, child_frame_id)
      br.sendTransform(tf_msg)

      # Sleep to control the rate
      rate.sleep()

if __name__ == '__main__':
  test_ros_visualization = TestRosVisualization()
  test_ros_visualization.run_test()
