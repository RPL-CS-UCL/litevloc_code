#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from map_manager import MapManager
from utils.utils_geom import convert_vec_to_matrix

def parse_arguments():
    parser = argparse.ArgumentParser(description='Map processing and visualization')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--scenes', type=str, required=True, nargs="+", help='Scenes')
    parser.add_argument('--viz', action='store_true', help='Visualize the graph')
    
    return parser.parse_args()

def plot_connected_cameras(poses, edge_list, title, ax, mode='2d'):
    x_coords, y_coords, z_coords = poses[:, 0], poses[:, 1], poses[:, 2]
    ax.plot(x_coords, y_coords, z_coords, 'o', color='g', label='Nodes', markersize=5)

    # Plot connections
    for edge in edge_list:
        node_id0, node_id1 = edge
        trans0 = poses[node_id0, :3]
        trans1 = poses[node_id1, :3]
        ax.plot([trans0[0], trans1[0]], [trans0[1], trans1[1]], [trans0[2], trans1[2]], '-', color='k', lw=1)
        
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    if mode == '2d':
        ax.view_init(elev=90, azim=90)
    elif mode == '3d':
        ax.view_init(elev=55, azim=60)
    ax.axis('equal')

def process_and_visualize_map(args):
    graph_configs = {
        'odom': {},
        'trav': {},
        'covis': {
            'resize': [512, 288],
            'depth_scale': 0.0,
            'load_rgb': False,
            'load_depth': False,
            'normalized': False,
            'color_correct': False,
        },
    }
    graph_types = ['covis', 'odom', 'trav']
    titles = ['Covisibility Graph', 'Odometry Graph', 'Traversability Graph']

    for scene in args.scenes:
        logging.info(f"Processing Scene {scene}")
        map_path = pathlib.Path(args.dataset_dir) / scene
        final_map = MapManager(map_path)
        final_map.init_graphs(graph_configs)
        final_map.load_graphs(graph_configs)
        print(str(final_map))

        # Visualize the three graphs
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
        for ax, graph_type, title in zip(axes, graph_types, titles):
            graph = final_map.graphs[graph_type]
            nodes = list(graph.nodes.values())
            if not nodes:
                ax.set_title(f"{title} (No Nodes)")
                continue
            
            # Sort nodes by ID
            node_ids = [node.id for node in graph.nodes.values()]
            poses = np.array([np.concatenate([node.trans, node.quat]) for node in graph.nodes.values()])
            
            # Extract edges
            edge_list = []
            for node in graph.nodes.values():
                for edge in node.edges.values():
                    nodeA_id = node.id
                    nodeB_id = edge[0].id
                    try:
                        idxA = node_ids.index(nodeA_id)
                        idxB = node_ids.index(nodeB_id)
                        edge_list.append([idxA, idxB])
                    except ValueError:
                        continue
            
            # Plot
            plot_connected_cameras(poses, edge_list, title, ax, mode='2d')
        
        plt.tight_layout()
        if args.viz:
            plt.show()
        else:
            plt.savefig(str(final_map.map_root / 'preds' / 'viz_graph.png'))

if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
    process_and_visualize_map(args)
