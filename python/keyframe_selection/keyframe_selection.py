#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import numpy as np
import argparse
from collections import OrderedDict
from utils.utils_geom import read_intrinsics, read_poses, read_timestamps
import pyiqa
import torch
import itertools
from pathlib import Path
import open3d as o3d

from estimator import get_estimator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Keyframe Selection Algorithm')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to the dataset')
    parser.add_argument('--scene', type=str, required=True, 
                       help='Scene name to process')
    return parser.parse_args()

class SubmapManager:
    def __init__(self, time_threshold=300.0):
        self.submaps = []
        self.current_submap = None
        self.time_threshold = time_threshold
        
    def add_frame(self, img_name, timestamp):
        if not self.submaps:
            self._create_new_submap(img_name, timestamp)
            return
            
        last_time = self.current_submap['end_time']
        if timestamp - last_time > self.time_threshold \
            or self.current_submap['frames'][0] == 'seq0/frame_00000.jpg':
            self._finalize_current_submap()
            self._create_new_submap(img_name, timestamp)
        else:
            self.current_submap['end_time'] = timestamp
            self.current_submap['frames'].append(img_name)
            
    def _create_new_submap(self, img_name, timestamp):
        self.current_submap = {
            'start_time': timestamp,
            'end_time': timestamp,
            'frames': [img_name]
        }
        self.submaps.append(self.current_submap)
        
    def _finalize_current_submap(self):
        if self.current_submap:
            self.current_submap['duration'] = \
                self.current_submap['end_time'] - self.current_submap['start_time']

def save_point_cloud(pts3d, save_path):
    """
    Save a point cloud to a file using Open3D.
    
    Args:
        pts3d (np.ndarray): Point cloud of shape [H, W, 3].
        save_path (str): Path to save the point cloud.
    """
    pts3d_flat = pts3d.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d_flat)
    o3d.io.write_point_cloud(save_path, pcd)
    return pcd

def overlap_ratio_compute(pcd1, pcd2):
    """
    Compute overlap_ratio and nonoverlap_ratio between two Open3D point clouds.
    
    Args:
        pcd1 (o3d.geometry.PointCloud): Point cloud A.
        pcd2 (o3d.geometry.PointCloud): Point cloud B.
    
    Returns:
        tuple: (overlap_ratio, nonoverlap_ratio)
    """
    def find_boundary(distances):
        sort_distances = sorted(distances)
        Q1 = np.percentile(sort_distances, 25)
        Q3 = np.percentile(sort_distances, 75)
        IQR = Q3 - Q1
        boundary = Q1 + 0.1 * IQR
        return boundary

    # Convert point clouds to NumPy arrays
    points1 = np.asarray(pcd1.points)
    kdtree_B = o3d.geometry.KDTreeFlann(pcd2)
    
    # Compute distances from points in pcd1 to their nearest neighbors in pcd2
    distances_A_to_B = [np.sqrt(kdtree_B.search_knn_vector_3d(point, 1)[2][0]) for point in points1]
    dis_thre = find_boundary(distances_A_to_B)
    
    overlap_ratio = np.sum([dist < dis_thre for dist in distances_A_to_B]) / len(points1)
    nonoverlap_ratio = 1.0 - overlap_ratio

    return overlap_ratio, nonoverlap_ratio

def pre_compute(scene_path):
    """Enhanced pre-computation with structured submap handling"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load raw data
    original_scene_path = scene_path.replace('keyframe_selection_eval', 'map_free_eval')
    intrinsics = read_intrinsics(os.path.join(original_scene_path, 'intrinsics.txt'))
    poses = read_poses(os.path.join(original_scene_path, 'poses.txt'))
    timestamps = read_timestamps(os.path.join(original_scene_path, 'timestamps.txt'))
    
    # Validate data consistency
    for key in poses.keys(): 
        if key not in timestamps:
            raise KeyError(f"{key} not found in timestamps")

    # Create submap split
    submap_mgr = SubmapManager()
    for img_name, timestamp in timestamps.items():
        submap_mgr.add_frame(img_name, timestamp[0])
    submap_mgr._finalize_current_submap()

    # Convert to structured numpy array
    submap_data = []
    for submap in submap_mgr.submaps:
        submap_data.append((
            submap['start_time'],
            submap['end_time'],
            np.array(submap['frames'])
        ))
    
    dtype = np.dtype([
        ('start_time', 'f8'), 
        ('end_time', 'f8'),
        ('frames', 'O')
    ])
    np.save(os.path.join(scene_path, 'submap_split.npy'), np.array(submap_data, dtype=dtype))
    print(f"{len(submap_data)} submaps are split")
       
    # Create iqa.txt
    IQA_METRIC = 'musiq'
    iqa_metric = pyiqa.create_metric(IQA_METRIC, device=device)
    iqa_scores = np.empty((len(poses), 2), dtype=object)
    for indice, (img_name, _) in enumerate(poses.items()):
        img_path = os.path.join(original_scene_path, img_name)
        score = iqa_metric(img_path).detach().squeeze(0).cpu().numpy()[0]
        iqa_scores[indice, 0], iqa_scores[indice, 1] = img_name, score
    np.savetxt(os.path.join(scene_path, 'iqa.txt'), iqa_scores, fmt="%s %.4f")

    # Create overlap matrix
    estimator = get_estimator('master', device)
    estimator.verbose = True
    output_dir = os.path.join(scene_path, 'preds')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries
    overlap_dict, nonoverlap_dict = {}, {}
    for img_name_0, img_name_1 in itertools.combinations(poses.keys(), 2):
        img_path_0 = os.path.join(original_scene_path, img_name_0)
        img_path_1 = os.path.join(original_scene_path, img_name_1)
        estimator(Path(scene_path), [img_path_0], img_path_1, None, None, None, dict())
        
        # Extract 3D points and masks
        pcd_list = []
        for i in range(2):
            conf = estimator.scene.get_masks()[i].cpu().numpy()
            pts3d = estimator.scene.get_pts3d()[i].detach().cpu().numpy()[conf]
            pcd = save_point_cloud(pts3d, os.path.join(output_dir, f'pts3d_{i}.pcd'))
            pcd_list.append(pcd)

        # Compute overlap and nonoverlap ratio        
        overlap_ratio, nonoverlap_ratio = overlap_ratio_compute(pcd_list[0], pcd_list[1])
        overlap_dict[(img_name_0, img_name_1)] = overlap_ratio
        nonoverlap_dict[(img_name_0, img_name_1)] = nonoverlap_ratio
        print(f'Overlap ratio: {overlap_ratio:.3f}, Nonoverlap ratio: {nonoverlap_ratio:.3f}')
        
        overlap_ratio, nonoverlap_ratio = overlap_ratio_compute(pcd_list[1], pcd_list[0])
        overlap_dict[(img_name_1, img_name_0)] = overlap_ratio
        nonoverlap_dict[(img_name_1, img_name_0)] = nonoverlap_ratio
        print(f'Overlap ratio: {overlap_ratio:.3f}, Nonoverlap ratio: {nonoverlap_ratio:.3f}')

    overlap_file = os.path.join(scene_path, 'overlap.npy')
    np.save(overlap_file, overlap_dict)
    print(f"Saved overlap_dict to {overlap_file}")

    nonoverlap_file = os.path.join(scene_path, 'nonoverlap.npy')
    np.save(nonoverlap_file, nonoverlap_dict)
    print(f"Saved nonoverlap_dict to {nonoverlap_file}")   

    # Copy timestamps
    import shutil
    timestamp_file = os.path.join(original_scene_path, 'timestamps.txt')
    destination_file = os.path.join(scene_path, 'timestamps.txt')
    shutil.copy2(timestamp_file, destination_file)
    print(f"Copied {timestamp_file} to {destination_file}")    

def load_scene_data(dataset_path, scene):
    """Improved data loading with submap structure conversion"""
    scene_path = os.path.join(dataset_path, scene)
    required_files = ['iqa.txt', 'overlap.npy', 'submap_split.npy', 'timestamps.txt']
    
    if not all(os.path.exists(os.path.join(scene_path, f)) for f in required_files):
        print("Pre-computed files not found. Computing...")
        pre_compute(scene_path)
    
    # Load and convert submap structure
    submap_array = np.load(os.path.join(scene_path, 'submap_split.npy'), allow_pickle=True)
    submap_splits = [{
        'start': item['start_time'],
        'end': item['end_time'],
        'frames': item['frames'].tolist()
    } for item in submap_array]
    
    return {
        'iqa_scores': np.load(os.path.join(scene_path, 'iqa.txt')),
        'overlap_dict': np.load(os.path.join(scene_path, 'overlap.npy')),
        'nonoverlap_dict': np.load(os.path.join(scene_path, 'nonoverlap.npy')),
        'submap_splits': submap_splits
    }

def main():
    args = parse_arguments()
    scene_data = load_scene_data(args.dataset_path, args.scene)
    
    # Add keyframe selection logic here using scene_data
    print(f"Loaded data with {len(scene_data['submap_splits'])} submaps")

if __name__ == "__main__":
    main()
