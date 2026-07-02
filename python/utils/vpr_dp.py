#!/usr/bin/env python

import argparse
import numpy as np
from matplotlib import pyplot as plt

if __package__:
    from .vpr_single_matching import PlaceRecognitionSingleMatching
else:
    from vpr_single_matching import PlaceRecognitionSingleMatching

class PlaceRecognitionGraphSearch(PlaceRecognitionSingleMatching):
	def __init__(self):
		super().__init__()
		
		self.MAX_DIST = 1.0 # Maximum distance for score calculation

		# Velocity parameters (expanded range)
		self.vMin = 0.6
		self.vMax = 3.0
		self.numVel = 20
		self.velocities = np.linspace(self.vMin, self.vMax, self.numVel).tolist()

		# DP parameters
		self.mu = 1
		self.cost_penalty = 0.7
		self.jump_len = 10 # Jump length for relocation sampling
		
	def initialize_model(self, db_descs):
		"""Initialize the model with database descs"""
		self.db_descs = db_descs
		
	def match(self, query_descs):
		"""
		Match query descriptors against database using graph-based approach
		"""
		D_all = self.compute_diff_matrix(query_descs)
		self.N, self.L = D_all.shape
		cost_matrix = np.power(D_all, self.mu)

		# Perform graph search
		layer_states = self._perform_graph_search(cost_matrix)
		final_j = len(layer_states) - 1
		final_states = layer_states[final_j]
		if not final_states:
			print("No complete path.")
			return [], None, None

		# Find best end state
		min_cost, best_state = np.inf, None
		for i_final, (cum_cost, _, _, _) in final_states.items():
			if cum_cost < min_cost:
				min_cost = cum_cost
				best_state = i_final
		best_score = self.MAX_DIST - min_cost / self.L

		# Backtrack
		path = []
		curr = best_state  # i at layer final_j
		for j in range(final_j, -1, -1):
			i_curr = curr
			path.append((i_curr, j))
			# get predecessor
			cum_cost, _, prev_i, prev_v = layer_states[j][i_curr]
			curr = prev_i  # for next iteration
		path.reverse()

		pred_db_query_rows = [(i, j) for i, j in path]

		return pred_db_query_rows, best_score
	
	def _perform_graph_search(self, cost_matrix):
		"""
		Perform graph search to find the best path
		"""
		column_indices = np.arange(0, self.N).tolist()
		layer_states = []

		# Initialize layer 0
		state0 = {}
		for i_coord in column_indices:
			cost = cost_matrix[i_coord, 0]
			state0[i_coord] = (cost, i_coord, None, None)
		layer_states.append(state0)

		# Iterate layers
		for j in range(0, self.L - 1):
			curr_states = layer_states[j]
			next_states = {}

			for i_coord, (cum_cost, i_cum, _, prev_v) in curr_states.items():
				if prev_v is None:
					vel_list = self.velocities
				else:
					vel_list = [prev_v]

				for v in vel_list:
					i_next = i_cum + v
					i_coord_next = int(i_next)
					if 0 <= i_coord_next < self.N:
						cost1 = cost_matrix[i_coord_next, j+1]
						new_cost = cum_cost + cost1
						key = i_coord_next
						prev_best = next_states.get(key, (np.inf, None, None, None))[0]
						if new_cost < prev_best:
							next_states[key] = (new_cost, i_next, i_coord, v) # keep the same velocity

				# NOTE(gogojjh): Relocation sampling by jumping to another sequence
				lower_i, upper_i = i_coord - self.jump_len, i_coord + self.jump_len
				for k in range(0, int(lower_i)):
					cost2 = cost_matrix[k, j+1]
					new_cost2 = cum_cost + cost2 + self.cost_penalty # regularization for not using sequence
					key2 = k
					prev_best2 = next_states.get(key2, (np.inf, None, None, None))[0]
					if new_cost2 < prev_best2:
						next_states[key2] = (new_cost2, k, i_coord, None) # keep the same velocity

				for k in range(int(upper_i), self.N):
					cost2 = cost_matrix[k, j+1]
					new_cost2 = cum_cost + cost2 + self.cost_penalty # regularization for not using sequence
					key2 = k
					prev_best2 = next_states.get(key2, (np.inf, None, None, None))[0]
					if new_cost2 < prev_best2:
						next_states[key2] = (new_cost2, k, i_coord, None) # keep the same velocity

			if not next_states:
				print(f"No surviving states at layer {j+1}, stopping early.")
				layer_states.append({})
				break

			layer_states.append(next_states)

		return layer_states

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("--db_map_path", type=str, help="Path to the database map file")
	# parser.add_argument("--query_map_path", type=str, help="Path to the query map file")
	args = parser.parse_args()
