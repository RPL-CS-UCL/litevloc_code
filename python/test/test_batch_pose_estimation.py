#! /usr/bin/env python

"""
Simplified script for parsing arguments, reading image graphs, and visualizing nodes with connected lines and text.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from image_graph import ImageGraphLoader as GraphLoader

from utils.utils_setting_color_font import acquire_color_palette
color_palette = acquire_color_palette()

import pycpptools.src.python.utils_math as pytool_math

from estimator import get_estimator, available_models
import torch
from PIL import Image

def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Image Graph Visualization")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset containing submaps.")
    parser.add_argument("--num_submap", type=int, required=True, help="Number of submaps to process.")
    parser.add_argument("--num_pairs", type=int, default=10, help="Number of random pairs to generate.")
    return parser.parse_args()

def read_image_graph(dataset_path, num_submap):
    """Load image graphs from the dataset."""
    submaps = []
    for i in range(num_submap):
        submap_path = os.path.join(dataset_path, f'out_map{i}')
        image_graph = GraphLoader.load_data(
            submap_path,
            [512, 288],
            depth_scale=0.0,
            load_rgb=True,
            load_depth=False,
            normalized=False
        )
        submaps.append(image_graph)
    return submaps

def visualize_submaps(submaps):
    """Visualize nodes with connected lines and node IDs."""
    plt.figure(figsize=(10, 8))
    for submap_idx, submap in enumerate(submaps):
        if submap_idx > 6: break
        submap = submaps[submap_idx]
        node_pos = np.array(
            [[node.trans_gt[0], node.trans_gt[1], node.trans_gt[2]] for node in submap.nodes.values()]
        )
        plt.plot(node_pos[:, 0], node_pos[:, 2], '-',
                 color=color_palette[min(submap_idx, len(color_palette))], 
                 label=f'Submap {submap_idx}')
        for node_id, node in submap.nodes.items():
            x, y, z = node.trans_gt[0], node.trans_gt[1], node.trans_gt[2]
            plt.text(x, z, str(node_id), fontsize=9, color='red')
        plt.legend()

    plt.title('Positions of Submap Nodes')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def create_random_pairs(submaps, num_pairs):
    random_pairs = []
    while len(random_pairs) < num_pairs:
        ref_submap_idx = np.random.randint(0, len(submaps))
        ref_submap = submaps[ref_submap_idx]
        ref_node_idx = np.random.randint(0, ref_submap.get_num_node())
        ref_node = ref_submap.nodes[ref_node_idx]

        tar_submap_idx = np.random.randint(0, len(submaps))
        tar_submap = submaps[tar_submap_idx]
        tar_node_idx = np.random.randint(0, tar_submap.get_num_node())
        tar_node = tar_submap.nodes[tar_node_idx]
        
        dis_trans, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
            ref_node.trans_gt, ref_node.quat_gt, tar_node.trans_gt, tar_node.quat_gt)
        if dis_trans > 3.0 and dis_trans < 10.0 and dis_angle < 150.0:
            random_pairs.append((ref_submap_idx, ref_node, tar_submap_idx, tar_node))

    return random_pairs

def run_pose_estimator(random_pairs, submaps, scene_root):
    est_opts = {
        'known_extrinsics': False,
        'known_intrinsics': False,
        'resize': 512,
    }    
    estimator = get_estimator("master", device='cuda', out_dir=Path('/Rocket_ssd/tmp/out_dir'))
    for pair in random_pairs:
        # ref_submap_id, ref_node, tar_submap_id, tar_node
        print(pair[0], pair[1].rgb_img_name, pair[2], pair[3].rgb_img_name)

        try:
            # Read the images
            # path_img0 = scene_root / f"out_map{pair[0]}" / pair[1].rgb_img_name
            # path_img1 = scene_root / f"out_map{pair[2]}" / pair[3].rgb_img_name
            # image0 = Image.open(path_img0)
            # image1 = Image.open(path_img1)
            # plt.figure(figsize=(10, 4))
            # plt.subplot(1, 2, 1)
            # plt.imshow(image0)
            # plt.title('Image 0')
            # plt.axis('off')
            # plt.subplot(1, 2, 2)
            # plt.imshow(image1)
            # plt.title('Image 1')
            # plt.axis('off')
            # plt.show()

            # Pose estimation
            list_img0_name = [f"out_map{pair[0]}/{pair[1].rgb_img_name}"]
            img1_name = f"out_map{pair[2]}/{pair[3].rgb_img_name}"
            result = estimator(scene_root, list_img0_name, img1_name, None, None, None, est_opts)
            print('Estimated pose: ', result['im_pose'][:3, 3:4].T) # Pose from world to camera
            estimator.show_reconstruction()
        except Exception as e:
            print(e)

if __name__ == '__main__':
    args = parse_arguments()
    submaps = read_image_graph(args.dataset_path, args.num_submap)
    # visualize_submaps(submaps)

    random_pairs = create_random_pairs(submaps, args.num_pairs)

    # run pose estimator
    run_pose_estimator(random_pairs, submaps, Path(args.dataset_path))
    
    
