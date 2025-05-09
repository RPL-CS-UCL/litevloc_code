"""
Usage:
python pose_fusion.py data_path /Rocket_ssd/dataset/data_litevloc/17DRP5sb8fy/
"""

import os
import gtsam
import argparse
import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path

from utils.utils_geom import convert_vec_gtsam_pose3
from utils.gtsam_pose_graph import PoseGraph

def parse_arguments():
	parser = argparse.ArgumentParser(description="Pose Fusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--isam_params", action="store_true", help="use ISAM2 specific parameters setting")
	parser.add_argument("--viz", action="store_true", help="visualize the result")
	parser.add_argument("--data_path", type=str, default="/tmp/", help="path to data")
	parser.add_argument("--odom_type", type=str, default="depth_reg", help="odometry type: depth_reg, leg_odom")
	args, unknown = parser.parse_known_args()
	return args	

class PoseFusion(PoseGraph):
	# The key difference from PoseGraph is the inclusion of timestamp information of each pose
	def __init__(self, args):
		super().__init__()
		self.timestamp = dict()
		if args.isam_params:
			params = gtsam.ISAM2Params()
			params.setRelinearizeThreshold(0.1)
			params.relinearizeSkip = 3
			self.isam = gtsam.ISAM2(params)
		else:
			self.isam = gtsam.ISAM2()

	def add_init_estimate(self, key: int, timestamp: float, pose: gtsam.Pose3):
		if self.initial_estimate.exists(key):
			self.initial_estimate.erase(key)
			self.initial_estimate.insert(key, pose)
		else:
			self.initial_estimate.insert(key, pose)
			self.timestamp[key] = timestamp

	def initalize_ros(self):
		self.pub_odom = rospy.Publisher('/pose_fusion/odometry', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/pose_fusion/path', Path, queue_size=10)
		self.pub_path_opt = rospy.Publisher('/pose_fusion/path_opt', Path, queue_size=10)
		
		self.path_msg = Path()
		self.path_msg_opt = Path()

def perform_pose_fusion(pose_fusion: PoseFusion, args):
	# Read data from file
	odometry_poses = np.loadtxt(os.path.join(args.data_path, 'poses_vo.txt'))
	vloc_poses = np.loadtxt(os.path.join(args.data_path, 'poses_vloc.txt'))
	gt_poses = np.loadtxt(os.path.join(args.data_path, 'poses_gt.txt'))

	# Simulate that only do pose fusion when global localization is available
	end_time = vloc_poses[-1, 0] + 20
	odometry_poses = odometry_poses[odometry_poses[:, 0] <= end_time, :]
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
			current_pose = current_pose * prev_odom.between(curr_odom)
		# the odometry is aligned with the global frame
		if init_system:
			pose_fusion.add_init_estimate(i, current_pose)
		else:
			pose_fusion.add_init_estimate(i, gtsam.Pose3())

		# Add prior factor
		for j in range(len(vloc_poses)):
			if abs(vloc_poses[j, 0] - odometry_poses[i, 0]) < 0.01: # VLOC pose is available
				pose3 = convert_vec_gtsam_pose3(vloc_poses[j, 1:4], vloc_poses[j, 4:])
				sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
				pose_fusion.add_prior_factor(i, pose3, sigma)
				pose_fusion.add_init_estimate(i, pose3)

				pose_fusion.perform_optimization()
				current_pose = pose_fusion.current_estimate.atPose3(i)
				init_system = True
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
				marginal_covariance = None
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
	pose_fusion.initalize_ros()

	perform_pose_fusion(pose_fusion, args)
