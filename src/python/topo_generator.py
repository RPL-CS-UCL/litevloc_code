# Usage1: python topo_generator.py \
#   --rgb_topic /habitat_camera/color/image \
#   --depth_topic /habitat_camera/depth/image \
#   --semantic_topic /habitat_camera/semantic/image \
#   --odometry_topic /habitat_camera/odometry \
#   --topo_int_trans 3.0 \
#   --topo_int_rot 45.0 \
#   --data_path /Rocket_ssd/dataset/data_cmu_navigation/cmu_navigation_matterport3d_17DRP5sb8fy \
#   --dataset_type matterport3d

# Usage2: python topo_generator.py \
#   --rgb_topic /rgb/image_rect_color/compressed \
#   --depth_topic /depth_to_rgb/hw_registered/image_rect_raw \
#   --odometry_topic /Odometry \
#   --imu_topic /sensors/imu \
#   --topo_int_trans 5.0 \
#   --topo_int_rot 45.0 \
#   --data_path /Rocket_ssd/dataset/data_anymal/anymal_real_message_ops_mos/anymal_real_message_ops_mos \
#   --dataset_type anymal

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
		# Initialize argument parser and add arguments
		parser = argparse.ArgumentParser(description="Camera data collector for synchronized RGBD images and poses.")
		parser.add_argument('--rgb_topic', type=str, required=True, help='Topic name for RGB images')
		parser.add_argument('--depth_topic', type=str, required=True, help='Topic name for depth images')
		parser.add_argument('--semantic_topic', type=str, required=True, help='Topic name for semantic images')
		parser.add_argument('--odometry_topic', type=str, required=True, help='Topic name for odometry data')
		parser.add_argument('--imu_topic', type=str, default='/imu', help='Topic name for IMU data')
		parser.add_argument('--topo_int_trans', type=float, default=3.0, help='Translation interval for topological map')
		parser.add_argument('--topo_int_rot', type=float, default=45.0, help='Rotation interval for topological map')
		parser.add_argument('--data_path', type=str, default='/tmp', help='Path to save data')
		parser.add_argument('--dataset_type', type=str, default='matterport3d', help='Type of dataset (matterport3d, anymal)')
		self.args = parser.parse_args()

		# Set image type and conversion function based on dataset type
		self.RGB_IMAGE_TYPE = CompressedImage if self.args.dataset_type == 'anymal' else Image
		self.RGB_CV_FUNCTION = CvBridge().compressed_imgmsg_to_cv2 if self.args.dataset_type == 'anymal' else CvBridge().imgmsg_to_cv2

		# Initialize ROS node
		rospy.init_node('topo_generator')

		# Setup subscribers and synchronizer for image and odometry topics
		rgb_sub = message_filters.Subscriber(self.args.rgb_topic, self.RGB_IMAGE_TYPE)
		depth_sub = message_filters.Subscriber(self.args.depth_topic, Image)
		semantic_sub = message_filters.Subscriber(self.args.semantic_topic, self.RGB_IMAGE_TYPE)
		camera_odom_sub = message_filters.Subscriber(self.args.odometry_topic, Odometry)
		ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, semantic_sub, camera_odom_sub], 100, 0.1, allow_headerless=True)
		ts.registerCallback(self.image_callback)

		# Setup IMU subscriber
		imu_sub = rospy.Subscriber(self.args.imu_topic, Imu, self.imu_callback)

		# Initialize TF listener
		self.tf_listener = TransformListener()

		# Initialize pose tracking variables
		self.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
		self.last_t = np.array([0.0, 0.0, 0.0])

		# Setup directories for saving data
		self.setup_directories()

		# Initialize frame and map IDs
		self.frame_id = 0
		self.map_id = 0

		# Open files for saving pose and IMU data
		self.pose_file = open(f'{self.args.data_path}/camera_pose.txt', 'a')
		self.noisy_pose_file = open(f'{self.args.data_path}/noisy_camera_pose.txt', 'a')
		self.topomap_pose_file = open(f'{self.args.data_path}/topo_map/camera_pose.txt', 'a')
		self.odom_pose_file = open(f'{self.args.data_path}/odom_pose.txt', 'a')
		self.imu_file = open(f'{self.args.data_path}/imu.txt', 'a')

		# Keep the node running
		rospy.spin()

	def image_callback(self, rgb_image, depth_image, semantic_image, camera_odom):
		print(f'image_callback: {self.frame_id}')
		timestamp = rgb_image.header.stamp

		# Convert odometry to translation and quaternion
		trans, quat = convert_rosodom_to_vec(camera_odom)
		self.save_pose(self.pose_file, timestamp, trans, quat)
		
		# Add Gaussian noise to pose and save
		noisy_trans, noisy_quat = add_gaussian_noise_to_pose(trans, quat)
		self.save_pose(self.noisy_pose_file, timestamp, noisy_trans, noisy_quat)

		# Convert and save RGB image
		cv_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
		cv2.imwrite(f'{self.args.data_path}/rgb_image/{self.frame_id:06d}.png', cv_image)

		# Convert and save depth image
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(depth_image, "passthrough")
		cv2.imwrite(f'{self.args.data_path}/depth_image/{self.frame_id:06d}.png', cv_image)

		# Convert and save semantic image
		cv_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")
		cv2.imwrite(f'{self.args.data_path}/semantic_image/{self.frame_id:06d}.png', cv_image)

		# Compute relative displacement and save to topological map if necessary
		dis_t, dis_angle = compute_relative_dis(self.last_t, self.last_quat, trans, quat)
		if dis_t > self.args.topo_int_trans or dis_angle > self.args.topo_int_rot:
			print(f'Save topo map: {dis_t:.3f}, {dis_angle:.3f}')
			self.save_pose(self.topomap_pose_file, timestamp, trans, quat)

			# Save RGB image for topological map
			cv_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
			cv2.imwrite(f'{self.args.data_path}/topo_map/rgb_image/{self.map_id:06d}.png', cv_image)

			# Save depth image for topological map
			cv_image = bridge.imgmsg_to_cv2(depth_image, "passthrough")
			cv2.imwrite(f'{self.args.data_path}/topo_map/depth_image/{self.map_id:06d}.png', cv_image)

			# Save semantic image for topological map
			cv_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")
			cv2.imwrite(f'{self.args.data_path}/topo_map/semantic_image/{self.map_id:06d}.png', cv_image)

			# Update last pose and map ID
			self.last_t, self.last_quat = trans, quat
			self.map_id += 1

		# Increment frame ID
		self.frame_id += 1

	def imu_callback(self, msg):
		line = f'{msg.header.stamp.to_sec():.9f} {msg.linear_acceleration.x:.3f} {msg.linear_acceleration.y:.3f} ' \
						f'{msg.linear_acceleration.z:.3f} {msg.angular_velocity.x:.3f} {msg.angular_velocity.y:.3f} ' \
						f'{msg.angular_velocity.z:.3f}\n'
		self.imu_file.write(line)
		print(f"Saved IMU at timestamp {msg.header.stamp}")

	def setup_directories(self):
		# Create necessary directories for saving data
		base_path = self.args.data_path
		paths = [
			base_path,
			f'{base_path}/rgb_image',
			f'{base_path}/depth_image',
			f'{base_path}/semantic_image',
			f'{base_path}/topo_map',
			f'{base_path}/topo_map/rgb_image',
			f'{base_path}/topo_map/depth_image',
			f'{base_path}/topo_map/semantic_image'
		]
		for path in paths:
				os.makedirs(path, exist_ok=True)

	def save_pose(self, file, timestamp, trans, rot):
		# Save pose to file: timestamp, x, y, z, qx, qy, qz, qw
		line = f'{timestamp.to_sec():.9f} {trans[0]:.3f} {trans[1]:.3f} {trans[2]:.3f} ' \
						f'{rot[1]:.3f} {rot[2]:.3f} {rot[3]:.3f} {rot[0]:.3f}\n'
		file.write(line)

if __name__ == '__main__':
	topo_generator = TopoGenerator()
	if rospy.is_shutdown():
		topo_generator.pose_file.close()
		topo_generator.topomap_pose_file.close()
		topo_generator.odom_pose_file.close()
		topo_generator.imu_file.close()
