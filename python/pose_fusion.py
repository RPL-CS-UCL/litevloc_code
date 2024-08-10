import os
import gtsam
import numpy as np
import rospy

from nav_msgs.msg import Odometry, Path
import argparse
import time

from pycpptools.src.python.utils_math.tools_eigen import convert_vec_gtsam_pose3

class PoseFusion:
	def __init__(self):
		"""Initialize gtsam factor graph and isam"""
		self.graph = gtsam.NonlinearFactorGraph()
		self.initail_estimate = gtsam.Values()
		self.current_estimate = gtsam.Values()

		# parameters = gtsam.ISAM2Params()
		# parameters.setRelinearizeThreshold(0.3)
		# parameters.relinearizeSkip = 5
		self.isam = gtsam.ISAM2()

	def add_prior_factor(self, key: int, pose: gtsam.Pose3):
		sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
		prior_cov = gtsam.noiseModel.Diagonal.Sigmas(sigma)		
		self.graph.add(gtsam.PriorFactorPose3(key, pose, prior_cov))

	def add_odometry_factor(self, 
							prev_key: int, prev_pose: gtsam.Pose3, 
							curr_key: int, curr_pose: gtsam.Pose3):
		delta_pose = prev_pose.between(curr_pose)
		sigma = np.array([np.deg2rad(5.), np.deg2rad(5.), np.deg2rad(5.), 0.05, 0.05, 0.01])
		odometry_cov = gtsam.noiseModel.Diagonal.Sigmas(sigma)		
		self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, delta_pose, odometry_cov))

	def add_init_estimate(self, key: int, pose: gtsam.Pose3):
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

	def publish_message(self):
		# Publish odometry results
		# TODO: Implement publishing logic
		pass

def perform_pose_fusion(pf: PoseFusion, args):
	# Read data from file
	poses_odom = np.loadtxt(os.path.join(args.data_path, 'poses_vo.txt'))
	poses_vloc = np.loadtxt(os.path.join(args.data_path, 'poses_vloc.txt'))
	poses_gt = np.loadtxt(os.path.join(args.data_path, 'poses_gt.txt'))

	# Simulate that only do pose fusion when global localization is available
	start_time = poses_vloc[0, 0]
	poses_odom = poses_odom[poses_odom[:, 0] >= start_time]
	poses_vloc = poses_vloc[poses_vloc[:, 0] >= start_time]
	poses_gt = poses_gt[poses_gt[:, 0] >= start_time]
	print(f"Number of odometry poses: {len(poses_odom)}")
	print(f"Number of VLOC poses: {len(poses_vloc)}")
	print(f"Number of ground truth poses: {len(poses_gt)}")

	# Perform pose fusion
	for i in range(len(poses_odom)):
		# Add odometry factor
		if i > 0:
			prev_pose = convert_vec_gtsam_pose3(poses_odom[i-1, 1:4], poses_odom[i-1, 4:])
			curr_pose = convert_vec_gtsam_pose3(poses_odom[i, 1:4], poses_odom[i, 4:])
			pf.add_odometry_factor(i - 1, prev_pose, i, curr_pose)
		# Add prior factor
		has_vloc = False
		for j in range(len(poses_vloc)):
			if abs(poses_vloc[j, 0] - poses_odom[i, 0]) < 0.01: # VLOC pose is available
				pose3 = convert_vec_gtsam_pose3(poses_vloc[j, 1:4], poses_vloc[j, 4:])
				pf.add_prior_factor(i, pose3)
				pf.add_init_estimate(i, pose3)
				start_time = time.time()
				pf.perform_optimization()
				print(f"Time taken for optimization: {time.time() - start_time:.6f} at pose {i}")
				has_vloc = True
				break
		if not has_vloc:
			pf.add_init_estimate(i, gtsam.Pose3())
	result = pf.perform_optimization()
	curr_est = result['current_estimate']

	# Evaluation
	gt_est = gtsam.Values()
	for i in range(len(poses_odom)):
		for j in range(len(poses_gt)):
			if abs(poses_gt[j, 0] - poses_odom[i, 0]) < 0.01:
				gt_est.insert(i, convert_vec_gtsam_pose3(poses_gt[j, 1:4], poses_gt[j, 4:]))
				break

	res_error = []
	for i in range(len(poses_odom)):
		if gt_est.exists(i) and curr_est.exists(i):
			res_error.append(np.linalg.norm(curr_est.atPose3(i).translation() - gt_est.atPose3(i).translation()))
	print(f"RMSE: {np.sqrt(np.mean(np.square(res_error))):.3f}")

	# Visualization
	import gtsam.utils.plot as gtsam_plot
	import matplotlib.pyplot as plt
	for i in range(len(poses_odom)):
		if curr_est.exists(i):
			marginal_cov = pf.get_margin_covariance(i)
			marginal_cov = None
			if marginal_cov is not None:
				gtsam_plot.plot_pose3(0, curr_est.atPose3(i), 1, marginal_cov)
			else:
				gtsam_plot.plot_pose3(0, curr_est.atPose3(i), 1)
	plt.axis('equal')

	for i in range(len(poses_odom)):
		if gt_est.exists(i):
			gtsam_plot.plot_pose3(1, gt_est.atPose3(i), 1, marginal_cov)
	plt.axis('equal')

	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Pose Fusion", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--data_path", type=str, default="/tmp/", help="path to data")
	args = parser.parse_args()

	rospy.init_node('pose_fusion', anonymous=True)
	pose_fusion = PoseFusion()
	pose_fusion.setup_ros_objects()

	perform_pose_fusion(pose_fusion, args)
