#! /usr/bin/env python

"""Visualizes camera poses from dataset sequences.

Usage:
python utils_visualization.py --dataset matterport3d/map_free_eval/test/ \
    --dataset_name Matterport3d --sample_rate 1
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from colorama import Fore
from utils.utils_geom import read_poses, convert_vec_to_matrix, convert_matrix_to_vec
from utils.utils_setting_color_font import acquire_color_palette
PALLETE = acquire_color_palette()

# Configure non-interactive backend if no display
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')

def _draw_orientation_arrow(ax, transform, length, style):
    """Draws orientation arrow for a single camera."""
    start = transform[:3, 3]
    direction = transform[:3, :3] @ np.array([0, 0, length])
    
    head_width = style['head_width']
    head_length = style['head_length']
    zorder = style['zorder']
    fc = style['fc']
    ax.arrow(start[0], start[2], direction[0], direction[2], 
             head_width=head_width*1.5, head_length=head_length*1.4,
             width=head_width*0.3, fc=fc, ec=fc, zorder=zorder)

def _configure_axes(ax, positions, padding):
    """Configures plot axes limits and appearance."""
    min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
    min_z, max_z = np.min(positions[:, 2]), np.max(positions[:, 2])
    
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_z - padding, max_z + padding)
    ax.set(xlabel='X [m]', ylabel='Z [m]', aspect='equal')
    ax.grid(ls='--', color='0.7')

def _parse_pose_row(row):
    """Parses pose data (map-free format) from text row."""
    quat = np.roll(row[:4], -1) # wxyz -> xyzw
    trans = row[4:]
    T_c2w = convert_vec_to_matrix(trans, quat)
    T_w2c = np.linalg.inv(T_c2w)
    trans, quat = convert_matrix_to_vec(T_w2c)
    return trans, quat

def plot_camera_poses(poses: np.ndarray, sample_rate: int, title: str) -> plt.Figure:
    """Plots 2D camera poses with orientation arrows.
    
    Args:
        poses: Array of camera poses (N x 7) containing translation and quaternion
        sample_rate: Subsampling rate for visualization
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    bounds = np.max(poses[:, :3], axis=0) - np.min(poses[:, :3], axis=0)
    max_bound = np.max(bounds) / 2
    arrow_length = max_bound / 10
    head_size = max_bound / 20

    sampled_poses = poses[::sample_rate, :]
    positions = []

    fig, ax = plt.subplots(figsize=(5, 5))
    for idx, pose in enumerate(sampled_poses):
        transform = convert_vec_to_matrix(pose[:3], pose[3:])
        positions.append(transform[:3, 3])
        arrow_style = {
            'head_width': head_size * (1.6 if idx == 0 else 1.0),
            'head_length': head_size * (1.6 if idx == 0 else 1.0),
            'fc': PALLETE[0] if idx == 0 else PALLETE[1],
            'zorder': 100 if idx == 0 else 0,
        }
        _draw_orientation_arrow(ax, transform, arrow_length, arrow_style)

    _configure_axes(ax, np.array(positions), max_bound/5)
    ax.set_title(title)
    return fig

def plot_camera_poses_pair(poses: np.ndarray, start_idx0, start_idx1, sample_rate: int, title: str) -> plt.Figure:
    """Plots 2D camera poses with orientation arrows.
    
    Args:
        poses1: Array of keyframe camera poses (N x 7) containing translation and quaternion
        poses2: Array of camera poses (N x 7) containing translation and quaternion
        sample_rate: Subsampling rate for visualization
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    bounds = np.max(poses[:, :3], axis=0) - np.min(poses[:, :3], axis=0)
    max_bound = np.max(bounds) / 2
    arrow_length = max_bound / 10
    head_size = max_bound / 20

    positions = []
    fig, ax = plt.subplots(figsize=(5, 5))
    for idx, pose in enumerate(poses[::sample_rate, :]):
        transform = convert_vec_to_matrix(pose[:3], pose[3:])
        positions.append(transform[:3, 3])
        arrow_style = {
            'head_width': head_size * 1.0,
            'head_length': head_size * 1.0,
            'fc': PALLETE[0] if idx < start_idx1 else PALLETE[1],
            'zorder': 100 if idx < start_idx1 else 0,
        }
        _draw_orientation_arrow(ax, transform, arrow_length, arrow_style)

    _configure_axes(ax, np.array(positions), max_bound/5)
    ax.set_title(title)
    return fig

def process_scene(scene_path: str, args: argparse.Namespace):
    """Processes a single scene directory."""
    poses_dict = read_poses(os.path.join(scene_path, 'poses.txt'))

    poses = []
    for key, pose in poses_dict.items():
        trans, quat = _parse_pose_row(pose)
        poses.append(np.concatenate((trans, quat)))

    title = f"{args.dataset_name}-{os.path.basename(scene_path)}-{len(poses)-1} frames"
    fig = plot_camera_poses(np.array(poses), args.sample_rate, title)

    output_dir = os.path.join(args.dataset, '../scene_stat')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{os.path.basename(scene_path)}_poses.pdf"))
    plt.close()
    
    return 1, len(poses) - 1  # (ref_count, query_count)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, 
                        help="Path to dataset directory")
    parser.add_argument("--dataset_name", required=True,
                        choices=["Matterport3d", "UCLCampus", "HKUSTGZCampus", "UCLCampusAria"],
                        help="Name of the dataset")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="Subsampling rate for visualization")
    return parser.parse_args()

def main():
    """Main processing pipeline."""
    args = parse_arguments()
    scene_dirs = sorted([
        os.path.join(args.dataset, d) 
        for d in os.listdir(args.dataset) 
        if os.path.isdir(os.path.join(args.dataset, d))
    ])

    ref_count = query_count = 0
    for scene_path in scene_dirs:
        print(Fore.GREEN + f'Processing {os.path.basename(scene_path)}...')
        ref, query = process_scene(scene_path, args)
        ref_count += ref
        query_count += query

    print(Fore.GREEN + f'Total {ref_count} reference scenes, {query_count} query images')

if __name__ == "__main__":
    main()
