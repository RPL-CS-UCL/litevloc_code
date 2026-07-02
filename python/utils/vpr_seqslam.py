#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

if __package__:
    from .vpr_single_matching import PlaceRecognitionSingleMatching
else:
    from vpr_single_matching import PlaceRecognitionSingleMatching

class PlaceRecognitionSeqMatching(PlaceRecognitionSingleMatching):
	def __init__(self, seqLen):
		super().__init__()

		# Base sequence length (used as reference)
		self.seqLen = seqLen
		self.matchWindow = 10
		self.MAX_DIST = 1.0
				
		# Velocity parameters (expanded range)
		self.vMin = 0.6
		self.vMax = 3.0
		self.numVel = 20
		
		# Original parameters remain
		self.wContrast = 10
		self.enhance = False

	def initialize_model(self, db_descs):
		"""Initialize the model with database descs"""
		self.db_descs = db_descs
		
	def match(self, query_descs, recall_values=1):
		"""
			Return:
				recall_preds: list of int, top recall values
				pred: int, best match
				score: float, score of the best match
		"""   
		if query_descs.shape[0] < self.seqLen:
			return self._fallback_match(query_descs[-1, :].reshape(1, -1), recall_values)

		D = self.compute_diff_matrix(query_descs)
		if self.enhance: 
			D = self._enhance_contrast(D)

		self.N, self.L = D.shape
		template_scores, template_velocities = \
			self._score_ref_templates(D, self.seqLen)
		recall_preds, pred, dist = \
			self._locate_best_match(template_scores, template_velocities, self.seqLen, recall_values)
		score = self.MAX_DIST - dist
		
		return recall_preds, pred, score

	def _fallback_match(self, query_desc, recall_values):
		"""Fallback to single-length matching if no good sequence is found"""
		dists = self.compute_dist_desc(query_desc)
		recall_preds = np.argsort(dists)[:recall_values]
		pred = recall_preds[0]
		score = self.MAX_DIST - dists[pred]
		
		return recall_preds, pred, score

	def _enhance_contrast(self, D):
		nref = D.shape[0]
		Denhanced = np.empty_like(D)
		for i in range(nref):
			# reference indices of window around each reference image
			idx_lower = max(i - int(self.wContrast / 2), 0)
			idx_upper = min(i + int(self.wContrast / 2) + 1, nref - 1)
			# local normalization of window given by indices above
			Denhanced[i, :] = (
				D[i, :] - np.mean(D[idx_lower:idx_upper, :], axis=0)
			) / np.std(D[idx_lower:idx_upper, :], axis=0)
		return Denhanced

	def _score_ref_templates(self, D, seq_len):
		# v = vMin, vMin+vStep, ..., vMax
		velocities = np.linspace(self.vMin, self.vMax, self.numVel + 1)
		# i = 0, ..., max_ind <- truncated so line search not cut off
		max_ind = int(self.N - 1 - self.vMax * seq_len)
		# last template image to begin sequence matching on: 0, 1, ..., max_ind - 1
		refs = np.arange(max_ind) 
		# D score for best velocity for each starting point (template image)
		# optD[i]: best score for sequence starting at template i
		optD = np.empty(max_ind); optD[:] = np.inf
		optV = np.empty(max_ind); optV[:] = np.inf

		# t = 0, ..., L
		times = np.arange(seq_len)
		for vel in velocities:
			# indices in D for line search given a particular velocity
			# include all template number
			row_indices = (
				np.floor(refs[:, np.newaxis] + vel * times[np.newaxis, :])
				.astype(int)
			) # a vector with dimension as max_ind x L
			# remove indices outside of D.shape[0]
			# row_indices[row_indices >= self.N] = self.N - 1
			row_indices = row_indices.reshape(-1)
			# line search indices for the query sequence
			col_indices = np.tile(times, max_ind)
			# evaluate D at indices and sum to get aggregate difference
			Dsum = np.sum(D[row_indices, col_indices].reshape(max_ind, seq_len), axis=1)
			# for sequence matching scores better than
			# prior scores (under different velocities), update
			ind_better = Dsum < optD
			optD[ind_better] = Dsum[ind_better]
			optV[ind_better] = vel

		return optD, optV

	def _locate_best_match(self, template_scores, template_velocities, seq_len, recall_values):
		# indices of best match and window around it
		iOpt = np.argmin(template_scores)
		iOptV = template_velocities[iOpt]
		iWinL = np.maximum(iOpt - int(self.matchWindow / 2), 0)
		iWinU = np.minimum(iOpt + int(self.matchWindow / 2), len(template_scores))
		# check best match outside window
		outside_scores = np.concatenate((template_scores[:iWinL], template_scores[iWinU:]))
		if np.any(outside_scores):
			optOutside = min(outside_scores)
			# for negative scores, u \in [0, 1]
			# increases the score... adjust
			if optOutside > 0:
				mu = template_scores[iOpt] / optOutside
			else:
				mu = optOutside / template_scores[iOpt]
		else:
			mu = template_scores[iOpt]

		pred = min(np.floor(iOpt + iOptV * (seq_len - 1)).astype(int), self.N - 1)
		indices = np.argsort(template_scores)[:recall_values]
		recall_preds = [min(self.N - 1, \
						np.floor(i + template_velocities[i] * (seq_len - 1)).astype(int)) \
						for i in indices]

		return recall_preds, pred, mu

if __name__ == "__main__":
	pass
	# import os
	# import sys
	# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
	# import argparse
	# from image_graph import ImageGraphLoader as GraphLoader
	# from tqdm import tqdm
	# import time

	# # Parse arguments
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--db_map_path", type=str, help="Path to the database map file")
	# parser.add_argument("--query_map_path", type=str, help="Path to the query map file")
	# args = parser.parse_args()

	# # Load database and query
	# db_map = GraphLoader.load_data(
	# 	args.db_map_path,
	# 	[512, 288],
	# 	depth_scale=0.0,
	# 	load_rgb=True,
	# 	load_depth=False,
	# 	normalized=False
	# )
	# query_map = GraphLoader.load_data(
	# 	args.query_map_path,
	# 	[512, 288],
	# 	depth_scale=0.0,
	# 	load_rgb=True,
	# 	load_depth=False,
	# 	normalized=False
	# )

	# # Extract descs
	# db_descs = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")
	# db_poses = np.zeros((db_map.get_num_node(), 7), dtype="float32")
	# for indices, (_, node) in enumerate(db_map.nodes.items()):
	# 	db_poses[indices, :3] = node.trans
	# 	db_poses[indices, 3:] = node.quat
	# query_descs = np.array([node.get_descriptor() for _, node in query_map.nodes.items()], dtype="float32")
	# query_poses = np.zeros((query_map.get_num_node(), 7), dtype="float32")
	# for indices, (_, node) in enumerate(query_map.nodes.items()):
	# 	query_poses[indices, :3] = node.trans
	# 	query_poses[indices, 3:] = node.quat

	# # Create sequence matching model
	# model = PlaceRecognitionSeqMatching(enable_ransac=False)
	# model.initialize_model(db_descs)

	# # Perform sequence matching
	# connected_indices = []
	# start_time = time.time()
	# for node in tqdm(query_map.nodes.values()):
	# 	query_descs = query_descs[max(0, node.id-model.seqLen+1) : node.id+1]
	# 	recall_preds, pred, score = model.match(query_descs, backward=False)
	# 	connected_indices.append((pred, node.id, score))
	# print(f"Sequence Matching Costs: {time.time() - start_time:.3f}s")

	# ################################################
	# # if model.ENABLE_RANSAC:
	# # 	D_all = model.compute_diff_matrix(query_descs)
	# # 	init_indices = connected_indices[:model.seqLen]
	# # 	best_indices, lines_coeff, cluster_data, cluster_labels = \
	# # 		model.ransac_check_match(D_all, connected_indices[int(model.seqLen/2):])
	# # 	best_indices += init_indices
	# # 	best_indices = list(dict.fromkeys(best_indices))
	# # else:
	# # 	best_indices = connected_indices

	# best_indices = connected_indices

	# ################################################ 
	# tp, tn, fp, fn = 0, 0, 0, 0
	# for edge in best_indices:
	# 	db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
	# 	dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
	# 		query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
	# 	)
	# 	if dis_tsl < 20.0:
	# 		tp += 1
	# 	else:
	# 		fp += 1
	# if tp + fp < 1:
	# 	precision = 0
	# else:
	# 	precision = tp / (tp+fp)
	# print(f"Precision: {precision:.3f} - {tp}/{tp+fp}")
	# ################################################

	# ################################################ Visualization
	# os.makedirs(f"{args.query_map_path}/preds", exist_ok=True)
	# fig, ax = plt.subplots(figsize=(10, 10))
	# for node_id, node in db_map.nodes.items():
	# 	ax.plot(node.trans_gt[0], node.trans_gt[1], 'ko', markersize=5)
	# 	for edge in node.edges.values():
	# 		next_node = edge[0]
	# 		ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)
	# for node_id, node in query_map.nodes.items():            
	# 	ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
	# 	for edge in node.edges.values():
	# 		next_node = edge[0]
	# 		ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)    
	# ax.grid(ls='--', color='0.7')
	# plt.axis('equal')
	# plt.xlabel('X-axis')
	# plt.ylabel('Y-axis')
	# plt.title(f"Precision: {precision:.3f} - {tp}/{tp+fp}")
	# plt.savefig(f"{args.query_map_path}/preds/result_PR.jpg")
	# plt.close()
	# ################################################

	# ################################################
	# model.save_diff_matrix_fitting(\
	# 	f"{args.query_map_path}/preds", 
	# 	connected_indices, best_indices, 
	# 	D_all, db_map, query_map, 
	# 	lines_coeff, cluster_data, cluster_labels)
	################################################
