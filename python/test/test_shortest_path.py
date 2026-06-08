'''
Usage1: python test_shortest_path.py --start_id 0 --goal_id 10 \
--depth_scale 0.001 --dataset_path /Titan/dataset/data_litevloc/anymal_ops_mos --sample_map 1
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import time
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as pl
pl.ion()

from pycpptools.src.python.utils_math.tools_eigen import compute_relative_dis
from pycpptools.src.python.utils_algorithm.shortest_path import dijk_shortest_path

from utils.utils_image_matching_method import *
from image_graph import ImageGraphLoader

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def main(args):
	"""Main function to run the image matching process."""
	image_size = args.image_size

	"""Load image data"""
	map_camera_type ='map_zed'
	path_map = os.path.join(args.dataset_path, map_camera_type)
	image_graph = ImageGraphLoader.load_data(path_map, image_size, depth_scale=args.depth_scale, normalized=False)
	print('Total number of nodes with IDs: ', image_graph.get_num_node(), image_graph.get_all_id())

	"""Create edges between nodes in the graph"""
	for map_id, _ in image_graph.nodes.items():
		if map_id == 0: 
			continue
		map_node_prev = image_graph.get_node(map_id - args.sample_map)
		map_node_next = image_graph.get_node(map_id)
		if (map_node_prev is not None) and (map_node_next is not None):
			# use the relative translation distance as the edge weight
			weight, _ = compute_relative_dis(map_node_prev.trans, map_node_next.quat, 
																 		   map_node_next.trans, map_node_next.quat,
																		   mode='xyzw')
			image_graph.add_edge_undirected(map_node_prev, map_node_next, weight)

	"""Perform shortest path searching"""
	start_time = time.time()
	start_node = image_graph.get_node(args.start_id)
	goal_node = image_graph.get_node(args.goal_id)
	travel_distance, path = dijk_shortest_path(image_graph, start_node, goal_node)
	out_str  = f"Time taken: {time.time() - start_time:.3f}s\n"
	out_str += f"Travel distance of the shortest path: {travel_distance:.3f}m\n"
	out_str += f"Shortest path: " + " -> ".join([str(node.id) for node in path])
	print(out_str)

if __name__ == "__main__":
	"""Setup command-line arguments."""
	parser = argparse.ArgumentParser(description="Shortest Path Test",
																	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--start_id", type=int, default=0, help="start_id")
	parser.add_argument("--goal_id", type=int, default=0, help="goal_id")
	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	parser.add_argument("--image_size", type=int, default=512, nargs="+",
											help="Resizing shape for images (HxW). If a single int is passed, set the"
											"smallest edge of all images to this value, while keeping aspect ratio")
	parser.add_argument('--depth_scale', type=float, default=0.001, help='habitat: 0.039, anymal: 0.001')
	parser.add_argument("--sample_map", type=int, default=1, help="sample of map")
	args = parser.parse_args()
	main(args)
