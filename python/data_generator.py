'''
Usage1: python data_generator.py \
--rgb_topic /habitat_camera/color/image \
--depth_topic /habitat_camera/depth/image \
--semantic_topic /habitat_camera/semantic/image \
--odometry_topic /habitat_camera/odometry \
--imu_topic /imu \
--topo_int_trans 3.0 \
--topo_int_rot 45.0 \
--data_path /Titan/dataset/data_topo_loc/cmu_navigation_matterport3d_17DRP5sb8fy \
--dataset_type matterport3d
'''
'''
Usage2: python data_generator.py \
--rgb_topic /rgb/image_rect_color/compressed \
--depth_topic /depth_to_rgb/hw_registered/image_rect_raw \
--semantic_topic /rgb/image_rect_color/compressed \
--odometry_topic /Odometry \
--imu_topic /imu \
--topo_int_trans 5.0 \
--topo_int_rot 45.0 \
--data_path /Titan/dataset/data_topo_loc/anymal_ops_mos \
--dataset_type anymal_vlp
'''  

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

from pycpptools.src.python.utils_ros.tools_ros_msg_conversion import convert_rosodom_to_vec
from pycpptools.src.python.utils_math.tools_eigen import add_gaussian_noise_to_pose, compute_relative_dis, convert_vec_to_matrix, convert_matrix_to_vec
from scipy.spatial.transform import Rotation

bridge = CvBridge()

class DataGenerator:
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
		parser.add_argument('--dataset_type', type=str, default='matterport3d', help='Type of dataset (matterport3d, anymal_vlp, anymal_livox)')
		self.args = parser.parse_args()

		# Set image type and conversion function based on dataset type
		self.RGB_IMAGE_TYPE = CompressedImage if 'anymal' in self.args.dataset_type else Image
		self.RGB_CV_FUNCTION = bridge.compressed_imgmsg_to_cv2 if 'anymal' in self.args.dataset_type else CvBridge().imgmsg_to_cv2

		# Initialize ROS node
		rospy.init_node('data_generator')

		# Setup subscribers and synchronizer for image and odometry topics
		rgb_sub = message_filters.Subscriber(self.args.rgb_topic, self.RGB_IMAGE_TYPE)
		depth_sub = message_filters.Subscriber(self.args.depth_topic, Image)
		semantic_sub = message_filters.Subscriber(self.args.semantic_topic, self.RGB_IMAGE_TYPE)
		base_odom_sub = message_filters.Subscriber(self.args.odometry_topic, Odometry)
		ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, semantic_sub, base_odom_sub], 100, 0.1, allow_headerless=True)
		ts.registerCallback(self.image_callback)

		# Setup IMU subscriber
		imu_sub = rospy.Subscriber(self.args.imu_topic, Imu, self.imu_callback)

		# Initialize TF listener
		# self.tf_listener = TransformListener()

		# Initialize pose tracking variables
		self.last_quat = np.array([1.0, 0.0, 0.0, 0.0])
		self.last_t = np.array([-1000.0, -1000.0, -1000.0])

		# NOTE(gogojjh): changed according to different dataset type, to transform the SLAM poses on the base_frame to the camera_frame
		if self.args.dataset_type =='matterport3d':
			self.T_base_cam = np.eye(4, 4)
		elif self.args.dataset_type == 'anymal_vlp':
			self.T_base_cam = convert_vec_to_matrix(
					np.array([-0.739, -0.056, -0.205]), 
					np.array([0.466, -0.469, -0.533, 0.528]),
					'xyzw')
		elif self.args.dataset_type == 'anymal_livox':
			self.T_base_cam = np.linalg.inv(convert_vec_to_matrix(
					np.array([-0.0509, -0.1229, 0.0047]), 
					np.array([0.4293, -0.4331, 0.5623, 0.5585]),
					'xyzw')
				)

		# Setup directories for saving data
		self.setup_directories()

		# Initialize camera pose, odom pose (simulated integrated IMU data), and imu measurements
		self.obs_camera_poses = np.empty((0, 8))
		self.obs_camera_poses_noisy = np.empty((0, 8))
		self.obs_odom_poses = np.empty((0, 8))
		self.obs_odom_poses_noisy = np.empty((0, 8))
		self.map_camera_poses = np.empty((0, 8))
		self.imu_measurements = np.empty((0, 7))

		# Keep the node running
		rospy.spin()

	def image_callback(self, rgb_image, depth_image, semantic_image, base_odom):
		print(f'image_callback: {self.obs_camera_poses.shape[0]}')
		timestamp = rgb_image.header.stamp

		# Convert and save RGB, depth, and semantic image
		cv_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
		cv2.imwrite(f'{self.args.data_path}/obs/rgb/{self.obs_camera_poses.shape[0]:06d}.png', cv_image)
		cv_image = bridge.imgmsg_to_cv2(depth_image, "passthrough")
		cv2.imwrite(f'{self.args.data_path}/obs/depth/{self.obs_camera_poses.shape[0]:06d}.png', cv_image)
		cv_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")
		cv2.imwrite(f'{self.args.data_path}/obs/semantic/{self.obs_camera_poses.shape[0]:06d}.png', cv_image)

		# Convert odometry to translation and quaternion
		trans, quat = convert_rosodom_to_vec(base_odom, 'xyzw')
		T_w_base = convert_vec_to_matrix(trans, quat, 'xyzw')
		T_w_cam = T_w_base @ self.T_base_cam
		trans, quat = convert_matrix_to_vec(T_w_cam, 'xyzw')
		# output format: timestamp, tx, ty, tz, qx, qy, qz, qw
		self.obs_camera_poses = np.vstack([self.obs_camera_poses, 
																		 np.array([timestamp.to_sec(), 
																			trans[0], trans[1], trans[2], 
																			quat[0], quat[1], quat[2], quat[3]])])
		self.obs_odom_poses = np.vstack([self.obs_odom_poses, 
																		 np.array([timestamp.to_sec(), 
																			trans[0], trans[1], trans[2], 
																			quat[0], quat[1], quat[2], quat[3]])])	
		
		noisy_trans, noisy_quat = add_gaussian_noise_to_pose(trans, quat, mode='xyzw')
		self.obs_camera_poses_noisy = np.vstack([self.obs_camera_poses_noisy, 
																		 np.array([timestamp.to_sec(), 
																			noisy_trans[0], noisy_trans[1], noisy_trans[2], 
																			noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3]])])
		self.obs_odom_poses_noisy = np.vstack([self.obs_odom_poses_noisy, 
																		 np.array([timestamp.to_sec(), 
																			noisy_trans[0], noisy_trans[1], noisy_trans[2], 
																			noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3]])])
		
		# Compute relative displacement and save to topological map if necessary
		dis_t, dis_angle = compute_relative_dis(self.last_t, self.last_quat, trans, quat, 'xyzw')
		if dis_t > self.args.topo_int_trans or dis_angle > self.args.topo_int_rot:
			print(f'Save map: dis_t: {dis_t:.3f}m, dis_angle: {dis_angle:.3f}deg')

			cv_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
			cv2.imwrite(f'{self.args.data_path}/map/rgb/{self.map_camera_poses.shape[0]:06d}.png', cv_image)
			cv_image = bridge.imgmsg_to_cv2(depth_image, "passthrough")
			cv2.imwrite(f'{self.args.data_path}/map/depth/{self.map_camera_poses.shape[0]:06d}.png', cv_image)
			cv_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")
			cv2.imwrite(f'{self.args.data_path}/map/semantic/{self.map_camera_poses.shape[0]:06d}.png', cv_image)
			
			self.map_camera_poses = np.vstack([self.map_camera_poses, 
																		 np.array([timestamp.to_sec(), 
																			trans[0], trans[1], trans[2], 
																			quat[0], quat[1], quat[2], quat[3]])])	
			
			# DEBUG(gogojjh):
			T_w_map0 = convert_vec_to_matrix(self.last_t, self.last_quat, 'xyzw')
			T_w_map1 = convert_vec_to_matrix(trans, quat, 'xyzw')
			T_map0_map1 = np.linalg.inv(T_w_map0) @ T_w_map1
			print(T_map0_map1)

			self.last_t, self.last_quat = trans, quat

	def imu_callback(self, msg):
		self.imu_measurements = np.vstack([self.imu_measurements, 
																		 np.array([msg.header.stamp.to_sec(),
																			msg.linear_acceleration.x,
																			msg.linear_acceleration.y,
																			msg.linear_acceleration.z,
																			msg.angular_velocity.x,
																			msg.angular_velocity.y,
																			msg.angular_velocity.z])])

	def setup_directories(self):
		base_path = self.args.data_path
		paths = [
			base_path,
			f'{base_path}/map',
			f'{base_path}/map/rgb',
			f'{base_path}/map/depth',
			f'{base_path}/map/semantic',
			f'{base_path}/obs',
			f'{base_path}/obs/rgb',
			f'{base_path}/obs/depth',
			f'{base_path}/obs/semantic',
		]
		for path in paths:
				os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
	data_generator = DataGenerator()
	if rospy.is_shutdown():
		print('ROS Shutdown, save poses to {}'.format(data_generator.args.data_path))
		np.savetxt(
			os.path.join(data_generator.args.data_path, 'obs', 'camera_pose_gt.txt'), 
			data_generator.obs_camera_poses, fmt='%.5f')
		np.savetxt(
			os.path.join(data_generator.args.data_path, 'obs', 'camera_pose_noisy.txt'), 
			data_generator.obs_camera_poses_noisy, fmt='%.5f')
		np.savetxt(
			os.path.join(data_generator.args.data_path, 'obs', 'odom_pose.txt'), 
			data_generator.obs_odom_poses, fmt='%.5f')
		np.savetxt(
			os.path.join(data_generator.args.data_path, 'obs', 'odom_pose_noisy.txt'), 
			data_generator.obs_odom_poses_noisy, fmt='%.5f')
		np.savetxt(
			os.path.join(data_generator.args.data_path, 'obs', 'imu_measurements.txt'), 
			data_generator.imu_measurements, fmt='%.5f')
		np.savetxt(
			os.path.join(data_generator.args.data_path, 'map', 'camera_pose_gt.txt'), 
			data_generator.map_camera_poses, fmt='%.5f')
