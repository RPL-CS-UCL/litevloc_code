import numpy as np
import gtsam
from typing import List, Optional, Tuple

class PoseGraph:
	def __init__(self):
		"""Initialize gtsam factor graph and isam"""
		self.graph = gtsam.NonlinearFactorGraph()
		self.initial_estimate = gtsam.Values()
		self.current_estimate = gtsam.Values()
		
		self.isam = gtsam.ISAM2()
		self.params = gtsam.ISAM2Params()

	def add_prior_factor(self, key: int, pose: gtsam.Pose3, sigma: np.ndarray) -> int:
		"""Add a prior factor and return its index in the factor graph."""
		noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigma)
		self.graph.add(gtsam.PriorFactorPose3(key, pose, noise_model))
		return self.graph.size() - 1

	def add_odometry_factor(self,
							prev_key: int, prev_pose: gtsam.Pose3,
							curr_key: int, curr_pose: gtsam.Pose3,
							sigma: np.ndarray) -> int:
		"""Add a relative pose factor and return its index in the factor graph."""
		noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigma)
		delta_pose = prev_pose.between(curr_pose)
		self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, delta_pose, noise_model))
		return self.graph.size() - 1

	def add_init_estimate(self, key: int, pose: gtsam.Pose3):
		if self.initial_estimate.exists(key):
			self.initial_estimate.erase(key)
			self.initial_estimate.insert(key, pose)
		else:
			self.initial_estimate.insert(key, pose)

	def perform_optimization(self):
		# graph_robust = PoseGraph.add_robust_kernel(graph)
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
	def add_robust_kernel(graph):
		graph_robust = gtsam.NonlinearFactorGraph()
		##### Huber robust kernel
		robust_model = gtsam.noiseModel.mEstimator.Huber.Create(k=1.345)
		##### Cauchy robust kernel
		# robust_model = gtsam.noiseModel.mEstimator.Cauchy.Create(k=0.5)
		#####
		for key in range(graph.size()):
			factor = graph.at(key)
			# TODO(gogojjh): Add robust kernel to othre factors
			if isinstance(factor, gtsam.BetweenFactorPose3):
				key1, key2 = factor.keys()
				noise_model = gtsam.noiseModel.Robust.Create(
					robust_model, factor.noiseModel()
				)
				new_factor = gtsam.BetweenFactorPose3(key1, key2, factor.measured(), noise_model)
				graph_robust.add(new_factor)
			else:
				graph_robust.add(factor.clone())
		
		return graph_robust

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
	def optimize_pose_graph_with_LM(graph, initial, verbose=False, robust_kernel=False):
		"""
		Optimizes a pose graph using the Levenberg-Marquardt algorithm.

		This function adds a prior factor to the first key in the initial estimate to anchor the graph,
		then optimizes the graph to minimize the error.

		Args:
			graph (gtsam.NonlinearFactorGraph): The pose graph containing factors (constraints).
			initial (gtsam.Values): Initial estimates for the variables (poses) in the graph.
			verbose (bool): Whether to print optimization progress.
			robust_kernel (bool): Whether to use a robust kernel for the optimization.

		Returns:
			gtsam.Values: The optimized values (poses) after the optimization process.
		"""    
		# Set up the optimizer
		params = gtsam.LevenbergMarquardtParams()
		if verbose:
			params.setVerbosity("Termination")
		
		if robust_kernel:
			graph_robust = PoseGraph.add_robust_kernel(graph)
			optimizer = gtsam.LevenbergMarquardtOptimizer(graph_robust, initial, params)
		else:
			optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
		
		result = optimizer.optimize()

		return result

	@staticmethod
	def optimize_pose_graph_with_GNC(
		graph: gtsam.NonlinearFactorGraph,
		initial: gtsam.Values,
		known_inlier_indices: Optional[List[int]] = None,
		loss: str = 'TLS',
		barc_prob: float = 0.99,
		verbose: bool = False,
	) -> Tuple[gtsam.Values, np.ndarray]:
		"""
		Optimizes a pose graph with Graduated Non-Convexity (GNC).

		Unlike a Huber kernel, the TLS loss is redescending: factors whose residual
		exceeds the inlier cost threshold get exactly zero weight instead of a
		saturated constant gradient that keeps pulling on the solution.

		Args:
			graph (gtsam.NonlinearFactorGraph): Factor graph built with NON-robust
				noise models. GNC is incompatible with gtsam.noiseModel.Robust.
			initial (gtsam.Values): Initial estimates for the variables (poses).
			known_inlier_indices (Optional[List[int]]): Factor indices exempt from
				outlier classification (odometry and prior factors). None means every
				factor is subject to classification.
			loss (str): 'TLS' (truncated least squares) or 'GM' (Geman-McClure).
			barc_prob (float): Chi-squared probability used to derive the inlier cost
				threshold. 0.99 corresponds to a threshold of 8.41 for Pose3.
			verbose (bool): Whether to print per-iteration GNC progress.

		Returns:
			Tuple[gtsam.Values, np.ndarray]: The optimized values, and a per-factor
			weight array of length graph.size(). A weight near 0 means the factor was
			classified as an outlier.
		"""
		loss_name = loss.upper()
		if loss_name == 'TLS':
			loss_type = gtsam.GncLossType.TLS
		elif loss_name == 'GM':
			loss_type = gtsam.GncLossType.GM
		else:
			raise ValueError(f"Unsupported GNC loss '{loss}', expected 'TLS' or 'GM'")

		gnc_params = gtsam.GncLMParams(gtsam.LevenbergMarquardtParams())
		gnc_params.setLossType(loss_type)
		if known_inlier_indices:
			gnc_params.setKnownInliers(list(known_inlier_indices))
		if verbose:
			gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.SUMMARY)

		optimizer = gtsam.GncLMOptimizer(graph, initial, gnc_params)
		optimizer.setInlierCostThresholdsAtProbability(barc_prob)
		result = optimizer.optimize()
		weights = np.asarray(optimizer.getWeights(), dtype=float)

		return result, weights

	@staticmethod
	def plot_pose_graph(save_dir, graph, results, titles, mode='2d', subgraph_keys=None):
		import os
		from matplotlib import pyplot as plt

		fig, axes = plt.subplots(1, len(results), subplot_kw={'projection': '3d'})		
		for ax, title, result in zip(axes, titles, results):
			resultPoses = gtsam.utilities.allPose3s(result)
			print(f"Number of resultPoses: {resultPoses.size()}")
			x_coords = [resultPoses.atPose3(i).translation()[0] for i in range(resultPoses.size())]
			y_coords = [resultPoses.atPose3(i).translation()[1] for i in range(resultPoses.size())]
			z_coords = [resultPoses.atPose3(i).translation()[2] for i in range(resultPoses.size())]
			ax.plot(x_coords, y_coords, z_coords, 'o', color='b', label='Est. Trajectory', markersize=3)

			for key in range(graph.size()):
				factor = graph.at(key)
				if isinstance(factor, gtsam.BetweenFactorPose3):
					key1, key2 = factor.keys()
					tsl1 = result.atPose3(key1).translation()
					tsl2 = result.atPose3(key2).translation()
					ax.plot([tsl1[0], tsl2[0]], [tsl1[1], tsl2[1]], [tsl1[2], tsl2[2]], '-', color='g', lw=1)

			if subgraph_keys is not None:
				for graph_id, keys in enumerate(subgraph_keys):
					tsl = result.atPose3(keys[0]).translation()
					ax.text(tsl[0], tsl[1], tsl[2], f'{graph_id}', fontsize=12, color='r', ha='center')

			# Set axis labels
			ax.set_xlabel('X [m]')
			ax.set_ylabel('Y [m]')
			ax.set_zlabel('Z [m]')
			# Set title
			ax.set_title(title)
			# Set view angle
			if mode == '2d':
				ax.view_init(elev=90, azim=90)
			elif mode == '3d':
				ax.view_init(elev=45, azim=60)
			ax.axis('equal')

		plt.tight_layout()
		if save_dir:
			plt.savefig(os.path.join(save_dir, 'pose_graph_refined.png'))
		else:
			plt.show()
