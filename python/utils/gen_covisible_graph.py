#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import faiss
import numpy as np
import time

from image_graph import ImageGraphLoader as GraphLoader
from utils_image_matching_method import initialize_img_matcher, save_visualization
from tqdm import tqdm

from pycpptools.src.python.utils_math.tools_eigen import compute_relative_dis

SUFF_EDGE_THRE = 100
EDGE_THRE_NOR = 300

def parse_args():
	parser = argparse.ArgumentParser(description='Generate covisible graph')
	parser.add_argument('--map_path', type=str, help='Path to the map')
	parser.add_argument('--matcher', type=str, default='sift', nargs='+', help='Image matching method')
	parser.add_argument('--device', type=str, default='cuda', help='Cuda or CPU')
	parser.add_argument('--n_kpts', type=int, default=2048, help='Number of keypoints')
	return parser.parse_args()

def generate_covisible_graph(args):
	# Create image graph
	image_graph = GraphLoader.load_data(
		args.map_path,
		[512, 288],
		depth_scale=0.0,
		load_rgb=True,
		load_depth=False,
		normalized=False,
		edge_type='odometry'
	)
	db_poses = np.empty((image_graph.get_num_node(), 7), dtype="float32")
	for indices, (_, node) in enumerate(image_graph.nodes.items()):
		db_poses[indices, :3] = node.trans
		db_poses[indices, 3:] = node.quat
	pose_faiss_index = faiss.IndexFlatL2(3)
	pose_faiss_index.add(db_poses[:, :3])

	odom_edge_str_set = {f"{node.id}_{edge[0].id}" for node in image_graph.nodes.values() 
    	                 for edge in node.edges if node.id < edge[0].id}
	for node in image_graph.nodes.values(): node.edges.clear()

	for matcher in args.matcher:
		# Initialize image matcher
		img_matcher = initialize_img_matcher(matcher, args.device, args.n_kpts)
		img_matcher.DEFAULT_RANSAC_ITERS = 50
		img_matcher.DEFAULT_RANSAC_CONF = 0.9
		img_matcher.DEFAULT_REPROJ_THRESH = 8

		# Iterate over each node and find nearest neighbors
		cnt, total_comp_time, total_inliers = 0, 0, 0
		edge_str_set = set()
		for node in tqdm(image_graph.nodes.values()):
			try:
				_, _, recall_preds = pose_faiss_index.range_search(node.trans.reshape(1, -1), 7.5**2)
				recall_preds = recall_preds[recall_preds != node.id]

				for pred in recall_preds:
					nei_node = image_graph.get_node(pred)
					# Remove depulicate computation
					if f"{node.id}_{nei_node.id}" in edge_str_set: continue
					if f"{nei_node.id}_{node.id}" in edge_str_set: continue
					# Remove unnecessary camera
					dis_tsl, dis_angle = compute_relative_dis(\
						node.trans, node.quat, \
						nei_node.trans, nei_node.quat)
					if dis_angle > 75.0: continue
					# Perform image matching				
					start_time = time.time()
					result = img_matcher(node.rgb_image, nei_node.rgb_image)
					total_comp_time += time.time() - start_time
					num_inliers, mkpts0, mkpts1 = (
						result["num_inliers"],
						result['matched_kpts0'],
						result['matched_kpts1']
					)
					if num_inliers > SUFF_EDGE_THRE:
						weight = min(num_inliers / EDGE_THRE_NOR, 1.0)
						node.edges.append((nei_node, weight))
						edge_str_set.add(f"{node.id}_{nei_node.id}")			
						nei_node.edges.append((node, weight))
						edge_str_set.add(f"{nei_node.id}_{node.id}")
						log_dir = os.path.join(args.map_path, "output_matcher")
						# _ = save_visualization(node.rgb_image, nei_node.rgb_image,
						# 	mkpts0, mkpts1, log_dir, int(len(edge_str_set)/2), n_viz=20)

					total_inliers += num_inliers
					cnt += 1
					# print(f"Node {node.id} and Node {nei_node.id} with inliers {num_inliers}")
			except:
				pass
		for pair in odom_edge_str_set - edge_str_set:
			node_id, nei_id = map(int, pair.split("_"))
			node, nei_node = image_graph.get_node(node_id), image_graph.get_node(nei_id)
			weight = 0.1
			node.edges.append((nei_node, weight))
			nei_node.edges.append((node, weight))
		
		print(f"Matching Methods: {matcher}:")
		print(f"Average computation time: {1000.0 * total_comp_time / cnt:.3f} ms.")
		print(f"Average number of inliers: {total_inliers / cnt:.3f}.")
		if len(args.matcher) == 1:
			cov_edge_path = os.path.join(args.map_path, f"covisible_edge_list.txt")
			image_graph.write_edge_list(cov_edge_path)

if __name__ == '__main__':
	import warnings
	warnings.filterwarnings("ignore", category=FutureWarning)

	args = parse_args()
	generate_covisible_graph(args)