import numpy as np
import gtsam

class PoseGraph:
	def __init__(self):
		"""Initialize gtsam factor graph and isam"""
		self.graph = gtsam.NonlinearFactorGraph()
		self.initial_estimate = gtsam.Values()
		self.current_estimate = gtsam.Values()
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
		if self.initial_estimate.exists(key):
			self.initial_estimate.erase(key)
			self.initial_estimate.insert(key, pose)
		else:
			self.initial_estimate.insert(key, pose)

	def perform_optimization(self):
		self.isam.update(self.graph, self.initial_estimate)
		self.current_estimate = self.isam.calculateEstimate()
		self.graph.resize(0)
		self.initial_estimate.clear()
		result = {'current_estimate': self.current_estimate}
		return result

	def get_margin_covariance(self, key: int):
		if self.current_estimate.exists(key):
			return self.isam.marginalCovariance(key)
		else:
			return None

	def get_factor_graph(self):
		return self.graph

	def get_initial_estimate(self):
		return self.initial_estimate

	def get_current_estimate(self):
		return self.current_estimate