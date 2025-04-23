#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import argparse
import logging
import pathlib
import numpy as np
import faiss
from tqdm import tqdm
from utils_image_matching_method import initialize_img_matcher
from map_manager import MapManager

from matching import available_models

##### Outdoor
# SUFF_EDGE_THRESH = 100
# EDGE_THRESH_NORM = 500
# MAX_DISTANCE = 5.0  # meters

##### Indoor
SUFF_EDGE_THRESH = 200
EDGE_THRESH_NORM = 500
MAX_DISTANCE = 3.5  # meters

def parse_arguments():
    parser = argparse.ArgumentParser(description='Map processing with covisibility edges')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--scenes', type=str, required=True, nargs="+", help='Scenes')
    parser.add_argument('--matcher', type=str, default='sift', help=f"{available_models}")
    parser.add_argument('--output', type=str, default=None, help='Output directory for processed map')
    parser.add_argument('--n_kpts', type=int, default=2048, help='Number of keypoints to extract')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for processing')
    return parser.parse_args()

def process_map(args):   
    # Load existing map
    graph_configs = {
        'odom': {},
        'trav': {},
        'covis': {
            'resize': [512, 288],
            'depth_scale': 0.0,
            'load_rgb': True,
            'load_depth': False,
            'normalized': False,
            'color_correct': False,
        },
    }

    for scene in args.scenes:      
        map_path = pathlib.Path(args.dataset_dir) / scene
        final_map = MapManager(map_path)
        final_map.init_graphs(graph_configs)
        final_map.load_graphs(graph_configs)
        print(f"Process Scene {scene}: {str(final_map)}")

        # Initialize image matcher
        img_matcher = initialize_img_matcher(args.matcher, args.device, args.n_kpts)
        img_matcher.ransac_iters = 1000
        # img_matcher.ransac_conf = 0.95
        # img_matcher.ransac_reproj_thresh = 5

        # Build position index for fast neighbor search
        node_ids = np.array([node.id for node in final_map.covis.nodes.values()], dtype='int64')
        positions = np.array([node.trans for node in final_map.covis.nodes.values()], dtype='float32')
        position_index = faiss.IndexFlatL2(3)
        position_index.add(positions)

        # Process each node
        edges_nodeAB_covis = []
        for node in tqdm(final_map.covis.nodes.values(), desc="Processing nodes"):
            # Find neighbors within radius
            _, distances, neighbor_indices = position_index.range_search(
                node.trans.reshape(1, -1), MAX_DISTANCE**2
            )
            
            # Check the covisibility between the reference node and quey nodes
            for row_idx in neighbor_indices:
                nei_node_idx = node_ids[row_idx]
                if nei_node_idx == node.id or nei_node_idx < node.id:
                    continue  # Skip self
                
                neighbor = final_map.covis.get_node(nei_node_idx)
                
                # Perform image matching
                try:
                    result = img_matcher(node.rgb_image, neighbor.rgb_image)
                except:
                    continue
                
                num_inliers = result["num_inliers"]
                
                # Calculate covisibility weight <= 1.0
                covis_weight = min(num_inliers / EDGE_THRESH_NORM, 1.0)
                
                # Add edges if thresholds met
                if num_inliers > SUFF_EDGE_THRESH or abs(node.id - neighbor.id) == 1:
                    edges_nodeAB_covis.append([node, neighbor, np.eye(4), num_inliers, covis_weight])

        weight_func1 = (lambda edge: edge[4])
        weight_func2 = (lambda edge: np.linalg.norm(edge[0].trans - edge[1].trans)) 
        for dst_graph_type, src_edges, weight_func in [
            ("covis", edges_nodeAB_covis, weight_func1),
            ("trav", edges_nodeAB_covis, weight_func2)
        ]:
            dst_edges = final_map.update_edges(src_edges, dst_graph_type)
            final_map.graphs[dst_graph_type].add_inter_edges(dst_edges, weight_func)

        # Save processed graph
        if args.output is not None:
            output_path = pathlib.Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)       
            final_map.covis.map_root = output_path
            final_map.trav.map_root = output_path

        final_map.covis.save_to_file(edge_only=True)
        final_map.trav.save_to_file(edge_only=True)

        logging.info(f"Saved processed map to {final_map.map_root}")

if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
    process_map(args)
