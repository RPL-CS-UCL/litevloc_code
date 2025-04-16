import numpy as np
import gtsam

class PoseGraph:
	def __init__(self):
		"""Initialize gtsam factor graph and isam"""
		self.graph = gtsam.NonlinearFactorGraph()
		self.initial_estimate = gtsam.Values()
		self.current_estimate = gtsam.Values()
		
		self.isam = gtsam.ISAM2()
		self.params = gtsam.ISAM2Params()

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
	
	@staticmethod
	def find_connected_components(graph):
		"""
		Find disconnected subgraphs using basic data structures.
		Return:
			components: [component1, component2, ...]
				component1: [key1, key2, ...] without sorting key id
		"""		
		# Build adjacency list using regular dict
		adjacency = {}
		all_keys = set()
		for key in range(graph.size()):
			factor = graph.at(key)
			if isinstance(factor, gtsam.BetweenFactorPose3):
				key1, key2 = factor.keys()
				if key1 not in adjacency:
					adjacency[key1] = set()
				if key2 not in adjacency:
					adjacency[key2] = set()
				adjacency[key1].add(key2)
				adjacency[key2].add(key1)
				
				all_keys.add(key1)
				all_keys.add(key2)

		visited = set()
		components = []
		# BFS implementation using list-as-queue
		for key in all_keys:
			if key not in visited:
				queue = [key]
				visited.add(key)
				component = []
				while queue:
					current = queue.pop()  # Dequeue from front
					component.append(current)
					if current in adjacency:
						for neighbor in adjacency[current]:
							if neighbor not in visited:
								visited.add(neighbor)
								queue.append(neighbor)
				
				components.append(component)

		return components
	@staticmethod
	def optimize_pose_graph_with_LM(graph, initial, verbose=False):
		"""
		Optimizes a pose graph using the Levenberg-Marquardt algorithm.

		This function adds a prior factor to the first key in the initial estimate to anchor the graph,
		then optimizes the graph to minimize the error.

		Args:
			graph (gtsam.NonlinearFactorGraph): The pose graph containing factors (constraints).
			initial (gtsam.Values): Initial estimates for the variables (poses) in the graph.

		Returns:
			gtsam.Values: The optimized values (poses) after the optimization process.
		"""    
		# Set up the optimizer
		params = gtsam.LevenbergMarquardtParams()
		if verbose:
			params.setVerbosity("Termination")
		optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
		result = optimizer.optimize()
		
		return result	

	@staticmethod
	def plot_pose_graph(save_dir, graph, result, mode='2d'):
		import os
		from matplotlib import pyplot as plt

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		resultPoses = gtsam.utilities.allPose3s(result)
		print(f"Number of resultPoses: {resultPoses.size()}")
		x_coords = [resultPoses.atPose3(i).translation()[0] for i in range(resultPoses.size())]
		y_coords = [resultPoses.atPose3(i).translation()[1] for i in range(resultPoses.size())]
		z_coords = [resultPoses.atPose3(i).translation()[2] for i in range(resultPoses.size())]
		plt.plot(x_coords, y_coords, z_coords, 'o', color='b', label='Est. Trajectory', markersize=3)

		for key in range(graph.size()):
			factor = graph.at(key)
			if isinstance(factor, gtsam.BetweenFactorPose3):
				key1, key2 = factor.keys()
				tsl1 = result.atPose3(key1).translation()
				tsl2 = result.atPose3(key2).translation()
				plt.plot([tsl1[0], tsl2[0]], [tsl1[1], tsl2[1]], [tsl1[2], tsl2[2]], '-', color='g', lw=1)

		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		ax.set_zlabel('Z [m]')
		if mode == '2d':
			ax.view_init(elev=90, azim=90)
		elif mode == '3d':
			ax.view_init(elev=55, azim=60)
		plt.tight_layout()
		plt.axis('equal')
		if save_dir:
			plt.savefig(os.path.join(save_dir, 'pose_graph_refined.png'))
		else:
			plt.show()
