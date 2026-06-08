#! /usr/bin/env python

import faiss
import torch
import numpy as np
from typing import Union
from matplotlib import pyplot as plt

class PlaceRecognitionSingleMatching:
	def __init__(self):
		self.seqLen = 1

	def initialize_model(self, db_descs):
		self.db_descs = db_descs
		self.db_faiss_index = faiss.IndexFlatL2(db_descs.shape[1])
		self.db_faiss_index.add(db_descs)

	def match(self, query_desc: np.ndarray, recall_values=1):
		_, recall_preds = self.db_faiss_index.search(query_desc, recall_values)
		
		dots = np.dot(self.db_descs[recall_preds[0][0], :], query_desc.T).reshape(-1)
		q_norms = np.linalg.norm(query_desc)
		db_norms = np.linalg.norm(self.db_descs[recall_preds[0][0], :])
		sims = dots / (q_norms * db_norms + 1e-8)
		score = sims

		return recall_preds[0], recall_preds[0][0], score

	def compute_dist_desc(self, query_desc: np.ndarray) -> np.ndarray:
		##### Option 1: cosine similarity
		dots = np.dot(self.db_descs, query_desc.T).reshape(-1)
		q_norm = np.linalg.norm(query_desc)
		db_norms = np.linalg.norm(self.db_descs, axis=1).reshape(-1)
		dists = 1.0 - dots / (q_norm * db_norms + 1e-8)
		##### Option 2: euclidean distance
		# dists = np.linalg.norm(self.db_descs - descriptor, axis=1)
		return dists

	def compute_diff_matrix(self, query_descs) -> np.ndarray:
		"""
			Return:
				D: np.ndarray, n_db x n_query
				query_descs: np.ndarray, n_query x descriptor_dim
		"""
		##### Option 1: cosine similarity
		dots = np.dot(self.db_descs, query_descs.T)
		db_norms = np.linalg.norm(self.db_descs, axis=1)[:, None]
		q_norms = np.linalg.norm(query_descs, axis=1)[None, :]
		D = 1.0 - dots / (db_norms * q_norms + 1e-8)
		##### Option 2: euclidean distance
		# D = np.linalg.norm(query_descs[None, :, :] - self.db_descs[:, None, :], axis=2)
		return D
	
	def viz_diff_matrix(self, save_img_path, D_all, db_query_rows=list()):
		plt.figure(figsize=(18, 9))
		plt.imshow(D_all, cmap='Greys', aspect='auto', clim=(0.0, 1.0))
		if len(db_query_rows) > 0:
			db_idx, query_idx = zip(*db_query_rows)
			plt.plot(query_idx, db_idx, 'g.', markersize=6, markeredgewidth=1)
		plt.colorbar()
		plt.xlabel('Query Index')
		plt.ylabel('Database Index')
		plt.title("Difference Matrix")
		plt.gca().set_aspect('equal')
		plt.tight_layout()
		plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
		plt.close()

if __name__ == "__main__":
	import os
	import sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
	import argparse
	from image_graph import ImageGraphLoader as GraphLoader
	from tqdm import tqdm

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--db_map_path", type=str, help="Path to the database map file")
	parser.add_argument("--query_map_path", type=str, help="Path to the query map file")
	args = parser.parse_args()
	# Load database and query
	db_map = GraphLoader.load_data(
		args.db_map_path,
		[512, 288],
		depth_scale=0.0,
		load_rgb=True,
		load_depth=False,
		normalized=False
	)
	query_map = GraphLoader.load_data(
		args.query_map_path,
		[512, 288],
		depth_scale=0.0,
		load_rgb=True,
		load_depth=False,
		normalized=False
	)
	# Performance test
	db_descriptors = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")
	model = PlaceRecognitionSingleMatching()
	model.initialize_model(db_descriptors, recall_values=5)
	preds = []
	for node in tqdm(query_map.nodes.values()):
		query_desc = node.get_descriptor()
		recall_preds, pred, score = model.match(query_desc.reshape(1, -1))
		preds.append(recall_preds)

	succ = 0
	for i, node in enumerate(query_map.nodes.values()):
		ref_map_node = db_map.nodes[preds[i][0]]
		dis_tsl, dis_angle = node.compute_distance(ref_map_node)
		if dis_tsl < 10.0 and dis_angle < 90.0:
			succ += 1
			print(f"Correct prediction: Query {node.id} - DB: {preds[i][0]}")
		else:
			print(f"Wrong prediction: Query {node.id} - DB: {preds[i][0]}")
	print(f"Success rate: {succ / len(query_map.nodes)}")