"""
Usage:
python pose_fusion.py \
--data_path /Rocket_ssd/dataset/data_topo_loc/17DRP5sb8fy/
"""

import os

import gtsam
import time
import argparse
import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path

from pycpptools.src.python.utils_math.tools_eigen import convert_vec_gtsam_pose3

def parse_arguments():
	parser = argparse.ArgumentParser(description="Pose Fusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--isam_params", action="store_true", help="use ISAM2 specific parameters setting")
	parser.add_argument("--viz", action="store_true", help="visualize the result")
	parser.add_argument("--data_path", type=str, default="/tmp/", help="path to data")
	args = parser.parse_args()
	return args	

class PoseFusion:
	def __init__(self, args):
		"""Initialize gtsam factor graph and isam"""
		self.graph = gtsam.NonlinearFactorGraph()
		self.initail_estimate = gtsam.Values()
		self.current_estimate = gtsam.Values()

		if args.isam_params:
			params = gtsam.ISAM2Params()
			params.setRelinearizeThreshold(0.1)
			params.relinearizeSkip = 3
			self.isam = gtsam.ISAM2(params)
		else:
			self.isam = gtsam.ISAM2()

	def add_prior_factor(self, key: int, pose: gtsam.Pose3, sigma: np.ndarray):
		prior_cov = gtsam.noiseModel.Diagonal.Sigmas(sigma)		
		self.graph.add(gtsam.PriorFactorPose3(key, pose, prior_cov))

	def add_odometry_factor(self, 
							prev_key: int, prev_pose: gtsam.Pose3, 
							curr_key: int, curr_pose: gtsam.Pose3, 
							sigma: np.ndarray):
		odometry_cov = gtsam.noiseModel.Diagonal.Sigmas(sigma)		
		delta_pose = prev_pose.between(curr_pose)
		self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, delta_pose, odometry_cov))

	def add_init_estimate(self, key: int, pose: gtsam.Pose3):
		if self.initail_estimate.exists(key):
			self.initail_estimate.erase(key)
		self.initail_estimate.insert(key, pose)

	def perform_optimization(self):
		self.isam.update(self.graph, self.initail_estimate)
		self.current_estimate = self.isam.calculateEstimate()
		self.graph.resize(0)
		self.initail_estimate.clear()
		result = {'current_estimate': self.current_estimate}
		return result

	def get_margin_covariance(self, key: int):
		if self.current_estimate.exists(key):
			return self.isam.marginalCovariance(key)
		else:
			return None

	def setup_ros_objects(self):
		self.pub_odom = rospy.Publisher('/pose_fusion/odometry', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/pose_fusion/path', Path, queue_size=10)
		self.path_msg = Path()

def perform_pose_fusion(pose_fusion: PoseFusion, args):
	# Read data from file
	odometry_poses = np.loadtxt(os.path.join(args.data_path, 'poses_vo.txt'))
	vloc_poses = np.loadtxt(os.path.join(args.data_path, 'poses_vloc.txt'))
	gt_poses = np.loadtxt(os.path.join(args.data_path, 'poses_gt.txt'))

	# Simulate that only do pose fusion when global localization is available
	start_time = vloc_poses[0, 0]
	end_time = vloc_poses[-1, 0]
	odometry_poses = odometry_poses[odometry_poses[:, 0] >= start_time, :]
	odometry_poses = odometry_poses[odometry_poses[:, 0] <= end_time, :]
	gt_poses = gt_poses[gt_poses[:, 0] >= start_time, :]
	gt_poses = gt_poses[gt_poses[:, 0] <= end_time, :]
	print(f"Number of odometry poses: {len(odometry_poses)}")
	print(f"Number of VLOC poses: {len(vloc_poses)}")
	print(f"Number of ground truth poses: {len(gt_poses)}")

	# Perform pose fusion
	current_pose = gtsam.Pose3()
	init_system = False
	for i in range(len(odometry_poses)):
		# Add odometry factor
		if i > 0:
			prev_odom = convert_vec_gtsam_pose3(odometry_poses[i-1, 1:4], odometry_poses[i-1, 4:])
			curr_odom = convert_vec_gtsam_pose3(odometry_poses[i, 1:4], odometry_poses[i, 4:])
			sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
			pose_fusion.add_odometry_factor(i - 1, prev_odom, i, curr_odom, sigma)
			if init_system:
				current_pose = current_pose * prev_odom.between(curr_odom)
				pose_fusion.add_init_estimate(i, current_pose)
		# Add prior factor
		for j in range(len(vloc_poses)):
			if abs(vloc_poses[j, 0] - odometry_poses[i, 0]) < 0.01: # VLOC pose is available
				pose3 = convert_vec_gtsam_pose3(vloc_poses[j, 1:4], vloc_poses[j, 4:])
				sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
				pose_fusion.add_prior_factor(i, pose3, sigma)
				pose_fusion.add_init_estimate(i, pose3)
				# start_time = time.time()
				pose_fusion.perform_optimization()
				current_pose = pose_fusion.current_estimate.atPose3(i)
				init_system = True
				# print(f"Time taken for optimization: {time.time() - start_time:.6f}s at pose {i}")
				break
	result = pose_fusion.perform_optimization()
	current_estimate = result['current_estimate']

	# Evaluation
	gt_estimate = gtsam.Values()
	for i in range(len(odometry_poses)):
		for j in range(len(gt_poses)):
			if abs(gt_poses[j, 0] - odometry_poses[i, 0]) < 0.01:
				gt_estimate.insert(i, convert_vec_gtsam_pose3(gt_poses[j, 1:4], gt_poses[j, 4:]))
				break

	residual_error = []
	for i in range(len(odometry_poses)):
		if gt_estimate.exists(i) and current_estimate.exists(i):
			residual_error.append(np.linalg.norm(current_estimate.atPose3(i).translation() - gt_estimate.atPose3(i).translation()))
	print(f"RMSE: {np.sqrt(np.mean(np.square(residual_error))):.3f}")

	# Visualization
	if args.viz:
		import gtsam.utils.plot as gtsam_plot
		import matplotlib.pyplot as plt
		for i in range(len(odometry_poses)):
			if current_estimate.exists(i):
				marginal_covariance = pose_fusion.get_margin_covariance(i)
				# marginal_covariance = None
				if marginal_covariance is not None:
					gtsam_plot.plot_pose3(0, current_estimate.atPose3(i), 1, marginal_covariance)
				else:
					gtsam_plot.plot_pose3(0, current_estimate.atPose3(i), 1)
		plt.title('Estimated poses')
		plt.axis('equal')

		for i in range(len(odometry_poses)):
			if gt_estimate.exists(i):
				gtsam_plot.plot_pose3(1, gt_estimate.atPose3(i), 1, marginal_covariance)
		plt.title('GT poses')
		plt.axis('equal')

		plt.show()

if __name__ == '__main__':
	args = parse_arguments()

	rospy.init_node('pose_fusion', anonymous=True)
	pose_fusion = PoseFusion(args)
	pose_fusion.setup_ros_objects()

	perform_pose_fusion(pose_fusion, args)
