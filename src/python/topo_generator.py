# Usage1: python topo_generator.py \
# 	--rgb_topic /habitat_camera/color/image \
# 	--depth_topic /habitat_camera/depth/image \
# 	--semantic_topic /habitat_camera/semantic/image \
# 	--odometry_topic /habitat_camera/odometry \
# 	--topo_int_trans 3.0 \
# 	--topo_int_rot 45.0 \
# 	--data_path /Rocket_ssd/dataset/data_cmu_navigation/cmu_navigation_matterport3d_17DRP5sb8fy
#   --dataset_type matterport3d

# Usage2: python topo_generator.py \
# 	--rgb_topic /rgb/image_rect_color/compressed \
# 	--depth_topic /depth_to_rgb/hw_registered/image_rect_raw \
# 	--odometry_topic /Odometry \
# 	--imu_topic /sensors/imu \
# 	--topo_int_trans 5.0 \
# 	--topo_int_rot 45.0 \
# 	--data_path /Rocket_ssd/dataset/data_anymal/anymal_real_message_ops_mos/anymal_real_message_ops_mos \
# 	--dataset_type anymal

import os
import argparse
import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf import TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CompressedImage
import math

# Initialize the node
rospy.init_node('topo_generator')

class TopoGenerator:
	def __init__(self):
		parser = argparse.ArgumentParser(description="Camera data collector for synchronized RGBD images and poses.")
		parser.add_argument('--rgb_topic', type=str, required=True, help='Topic name for RGB images')
		parser.add_argument('--depth_topic', type=str, required=True, help='Topic name for depth images')
		parser.add_argument('--semantic_topic', type=str, required=True, help='Topic name for semantic images')
		parser.add_argument('--odometry_topic', type=str, required=True, help='Topic name for odometry data')
		parser.add_argument('--imu_topic', type=str, default='/imu', required=False, help='Topic name for IMU data')
		parser.add_argument('--topo_int_trans', type=float, default=3.0)
		parser.add_argument('--topo_int_rot', type=float, default=45.0)
		parser.add_argument('--data_path', type=str, default='/tmp', help='data_path')
		parser.add_argument('--dataset_type', type=str, default='matterport3d', help='matterport3d, anymal')	
		self.args = parser.parse_args()

		# ROS Subscriber
		if self.args.dataset_type == 'anymal':
			rgb_sub = message_filters.Subscriber(self.args.rgb_topic, CompressedImage)
			depth_sub = message_filters.Subscriber(self.args.depth_topic, CompressedImage)
			semantic_sub = message_filters.Subscriber(self.args.semantic_topic, Image)
			camera_odom_sub = message_filters.Subscriber(self.args.odometry_topic, Odometry)
			ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, semantic_sub, camera_odom_sub], 100, 0.1, allow_headerless=True)
			ts.registerCallback(self.image_callback)
		else:
			rgb_sub = message_filters.Subscriber(self.args.rgb_topic, Image)
			depth_sub = message_filters.Subscriber(self.args.depth_topic, Image)
			semantic_sub = message_filters.Subscriber(self.args.semantic_topic, Image)
			camera_odom_sub = message_filters.Subscriber(self.args.odometry_topic, Odometry)
			ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, semantic_sub, camera_odom_sub], 100, 0.1, allow_headerless=True)
			ts.registerCallback(self.image_callback)

		imu_sub = rospy.Subscriber(self.args.imu_topic, Imu, self.imu_callback)

		# Initialize TF buffer and listener
		self.tf_listener = TransformListener()

		self.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
		self.last_t = np.array([0.0, 0.0, 0.0])

		# Create folder
		self.setup_directories()

		# Parameter
		self.frame_id = 0
		self.map_id = 0
		self.pose_file = open('{}/camera_pose.txt'.format(self.args.data_path), 'a')
		self.noisy_pose_file = open('{}/noisy_camera_pose.txt'.format(self.args.data_path), 'a')
		self.topomap_pose_file = open('{}/topo_map/camera_pose.txt'.format(self.args.data_path), 'a')
		self.odom_pose_file = open('{}/odom_pose.txt'.format(self.args.data_path), 'a')
		self.imu_file = open('{}/imu.txt'.format(self.args.data_path), 'a')

		rospy.spin()

	def image_callback(self, rgb_image, depth_image, semantic_image, camera_odom):
		print('image_callback: {}'.format(self.frame_id))
		timestamp = rgb_image.header.stamp

		trans = np.array([camera_odom.pose.pose.position.x, 
											camera_odom.pose.pose.position.y, 
											camera_odom.pose.pose.position.z])
		rot = np.array([camera_odom.pose.pose.orientation.w, 
										camera_odom.pose.pose.orientation.x,
										camera_odom.pose.pose.orientation.y,
										camera_odom.pose.pose.orientation.z])			
		self.save_pose(self.pose_file, timestamp, trans, rot)
		
		noisy_trans, noisy_rot = self.add_pose_noise(trans, rot)
		self.save_pose(self.noisy_pose_file, timestamp, noisy_trans, noisy_rot)

		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(rgb_image, "bgr8")
		cv2.imwrite('{}/rgb_image/{:06d}.png'.format(self.args.data_path, self.frame_id), cv_image)
		cv_image = bridge.imgmsg_to_cv2(depth_image, "mono8")
		cv2.imwrite('{}/depth_image/{:06d}.png'.format(self.args.data_path, self.frame_id), cv_image)
		cv_image = bridge.imgmsg_to_cv2(semantic_image, "bgr8")
		cv2.imwrite('{}/semantic_image/{:06d}.png'.format(self.args.data_path, self.frame_id), cv_image)

		curr_t = np.array([trans[0], trans[1], trans[2]])
		curr_quat = np.array([rot[0], rot[1], rot[2], rot[3]])
		rel_t, rel_angle = self.compute_relative_dis(self.last_t, self.last_quat, curr_t, curr_quat)
		if rel_t > self.args.topo_int_trans or rel_angle > self.args.topo_int_rot:
			print('Save topo map: {:3f}, {:3f}'.format(rel_t, rel_angle))
			self.save_pose(self.topomap_pose_file, timestamp, trans, rot)
			cv_image = bridge.imgmsg_to_cv2(rgb_image, "bgr8")
			cv2.imwrite('{}/topo_map/rgb_image/{:06d}.png'.format(self.args.data_path, self.map_id), cv_image)
			cv_image = bridge.imgmsg_to_cv2(depth_image, "mono8")
			cv2.imwrite('{}/topo_map/depth_image/{:06d}.png'.format(self.args.data_path, self.map_id), cv_image)
			cv_image = bridge.imgmsg_to_cv2(semantic_image, "bgr8")
			cv2.imwrite('{}/topo_map/semantic_image/{:06d}.png'.format(self.args.data_path, self.map_id), cv_image)
			self.last_t, self.last_quat = curr_t, curr_quat
			self.map_id += 1

		self.frame_id += 1

	def odom_callback(self, msg):
		pass

	def imu_callback(self, msg):
		line = '{:9f} {:3f} {:3f} {:3f} {:3f} {:3f} {:3f}\n'.format(
			msg.header.stamp.to_sec(), 
			msg.linear_acceleration.x,
			msg.linear_acceleration.y,
			msg.linear_acceleration.z,
			msg.angular_velocity.x,
			msg.angular_velocity.y,
			msg.angular_velocity.z
		)
		self.imu_file.write(line)
		print(f"Saved IMU at timestamp {msg.header.stamp}")

	def setup_directories(self):
		os.makedirs('{}'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/rgb_image'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/depth_image'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/semantic_image'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/topo_map'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/topo_map/rgb_image'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/topo_map/depth_image'.format(self.args.data_path), exist_ok=True)
		os.makedirs('{}/topo_map/semantic_image'.format(self.args.data_path), exist_ok=True)

	def compute_relative_dis(self, last_t, last_quat, curr_t, curr_quat):
		rot1 = R.from_quat(np.roll(last_quat, -1)) # [qw qx qy qz] -> [qx qy qz qw]
		rot2 = R.from_quat(np.roll(curr_quat, -1))   
		rel_rot = rot2 * rot1.inv()
		rel_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
		rel_trans = np.linalg.norm(rot1.inv().apply(last_t - curr_t))
		return rel_trans, rel_angle

	def add_pose_noise(self, trans, quat, mean=0, stddev=0.1):
		noisy_trans = [x + np.random.normal(mean, stddev) for x in trans]
		rot = R.from_quat(np.roll(quat, -1))

		euler = rot.as_euler('xyz', degrees=False)
		noisy_euler = [x + np.random.normal(mean, stddev) for x in euler]
		noisy_rot = R.from_euler('xyz', noisy_euler, degrees=False)
		noisy_quat = np.roll(noisy_rot.as_quat(), 1)
		return noisy_trans, noisy_quat
	
	def save_pose(self, file, timestamp, trans, rot):
		line = '{:9f} {:3f} {:3f} {:3f} {:3f} {:3f} {:3f}\n'.format(
			timestamp.to_sec(), trans[0], trans[1], trans[2], rot[1], rot[2], rot[3], rot[0])
		file.write(line)

if __name__ == '__main__':
	topo_generator = TopoGenerator()
	if rospy.is_shutdown():
		topo_generator.pose_file.close()
		topo_generator.topomap_pose_file.close()
		topo_generator.odom_pose_file.close()		
		topo_generator.imu_file.close()
