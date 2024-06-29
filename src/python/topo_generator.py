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
import numpy as np

import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf import TransformListener
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CompressedImage

from pycpptools.python.utils_ros.tools_ros_msg_conversion import convert_rosodom_to_vec
from pycpptools.python.utils.ros.tools_eigen import add_gaussian_noise_to_pose, compute_relative_dis

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

		if self.args.dataset_type == 'anymal':
			self.RGB_IMAGE_TYPE = CompressedImage
			self.RGB_CV_FUNCTION = CvBridge().compressed_imgmsg_to_cv2
		else:
			self.RGB_IMAGE_TYPE = Image
			self.RGB_CV_FUNCTION = CvBridge().imgmsg_to_cv2

		# Initialize the node
		rospy.init_node('topo_generator')

		# ROS Subscriber
		rgb_sub = message_filters.Subscriber(self.args.rgb_topic, self.RGB_IMAGE_TYPE)
		depth_sub = message_filters.Subscriber(self.args.depth_topic, Image)
		semantic_sub = message_filters.Subscriber(self.args.semantic_topic, self.RGB_IMAGE_TYPE)
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

		# Files
		self.pose_file = open('{}/camera_pose.txt'.format(self.args.data_path), 'a')
		self.noisy_pose_file = open('{}/noisy_camera_pose.txt'.format(self.args.data_path), 'a')
		self.topomap_pose_file = open('{}/topo_map/camera_pose.txt'.format(self.args.data_path), 'a')
		self.odom_pose_file = open('{}/odom_pose.txt'.format(self.args.data_path), 'a')
		self.imu_file = open('{}/imu.txt'.format(self.args.data_path), 'a')

		rospy.spin()

	def image_callback(self, rgb_image, depth_image, semantic_image, camera_odom):
		print('image_callback: {}'.format(self.frame_id))
		timestamp = rgb_image.header.stamp

		trans, quat = convert_rosodom_to_vec(camera_odom)
		self.save_pose(self.pose_file, timestamp, trans, quat)
		
		noisy_trans, noisy_quat = add_gaussian_noise_to_pose(trans, quat)
		self.save_pose(self.noisy_pose_file, timestamp, noisy_trans, noisy_quat)

		bridge = CvBridge()
		cv_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
		cv2.imwrite('{}/rgb_image/{:06d}.png'.format(self.args.data_path, self.frame_id), cv_image)

		cv_image = bridge.imgmsg_to_cv2(depth_image, "mono8")
		cv2.imwrite('{}/depth_image/{:06d}.png'.format(self.args.data_path, self.frame_id), cv_image)

		cv_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")
		cv2.imwrite('{}/semantic_image/{:06d}.png'.format(self.args.data_path, self.frame_id), cv_image)

		dis_t, dis_angle = compute_relative_dis(self.last_t, self.last_quat, trans, quat)
		if dis_t > self.args.topo_int_trans or dis_angle > self.args.topo_int_quat:
			print('Save topo map: {:3f}, {:3f}'.format(dis_t, dis_angle))
			self.save_pose(self.topomap_pose_file, timestamp, trans, quat)

			cv_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
			cv2.imwrite('{}/topo_map/rgb_image/{:06d}.png'.format(self.args.data_path, self.map_id), cv_image)

			cv_image = bridge.imgmsg_to_cv2(depth_image, "mono8")
			cv2.imwrite('{}/topo_map/depth_image/{:06d}.png'.format(self.args.data_path, self.map_id), cv_image)

			cv_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")
			cv2.imwrite('{}/topo_map/semantic_image/{:06d}.png'.format(self.args.data_path, self.map_id), cv_image)

			self.last_t, self.last_quat = trans, quat
			self.map_id += 1

		self.frame_id += 1

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
