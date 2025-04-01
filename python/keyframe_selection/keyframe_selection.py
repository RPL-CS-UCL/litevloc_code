#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import numpy as np
import argparse
from collections import OrderedDict
import pyiqa
import torch
import itertools
from pathlib import Path
import open3d as o3d
from sklearn.mixture import GaussianMixture

from utils.utils_geom import read_intrinsics, read_poses, read_timestamps, read_descriptors
from utils.utils_vpr_method import *
import torchvision.transforms as transforms
from PIL import Image

from estimator import get_estimator

from landmark_selector import LandmarkSelector

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def edge_str(i, j):
    return f'{i}_{j}'


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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Keyframe Selection Algorithm')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to the dataset')
    parser.add_argument('--scene', type=str, required=True, 
                       help='Scene name to process')
    parser.add_argument('--method', type=str, required=True, 
                       help='landmark, random')
    return parser.parse_args()

def save_point_cloud(pts3d, save_path, save_flag=False):
    """
    Save a point cloud to a file using Open3D.
    
    Args:
        pts3d (np.ndarray): Point cloud of shape [H, W, 3].
        save_path (str): Path to save the point cloud.
    """
    pts3d_flat = pts3d.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d_flat)
    if save_flag:
        o3d.io.write_point_cloud(save_path, pcd)
    return pcd

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
    # IQA_METRIC = 'musiq'
    # iqa_metric = pyiqa.create_metric(IQA_METRIC, device=device)
    # iqa_scores = np.empty((len(poses), 2), dtype=object)
    # for indice, (img_name, _) in enumerate(poses.items()):
    #     img_path = os.path.join(original_scene_path, img_name)
    #     score = iqa_metric(img_path).detach().squeeze(0).cpu().numpy()[0]
    #     iqa_scores[indice, 0], iqa_scores[indice, 1] = img_name, score
    # np.savetxt(os.path.join(scene_path, 'iqa.txt'), iqa_scores, fmt="%s %.4f")

    # # Create descriptor.txt
    # desc_dimenson = 256
    # vpr_model = initialize_vpr_model('cosplace', 'ResNet18', desc_dimenson, device)    
    # transformations = [
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Resize(size=[512, 288], antialias=True)
    # ]
    # transform = transforms.Compose(transformations)
    # all_descriptors = np.empty((len(poses), desc_dimenson + 1), dtype=object)
    # for indice, (img_name, _) in enumerate(poses.items()):
    #     img_path = os.path.join(original_scene_path, img_name)
    #     pil_img = Image.open(img_path).convert("RGB")
    #     normalized_img = transform(pil_img)
    #     descriptors = vpr_model(normalized_img.unsqueeze(0).to(device))
    #     descriptors = descriptors.detach().cpu().numpy()
    #     all_descriptors[indice, 0], all_descriptors[indice, 1:] = img_name, descriptors
    # np.savetxt(os.path.join(scene_path, 'descriptors.txt'), all_descriptors, fmt="%s" + " %.9f" * desc_dimenson)

    # Create information reduction and information gain
    estimator = get_estimator('master', device)
    estimator.verbose = True
    output_dir = os.path.join(scene_path, 'preds')
    os.makedirs(output_dir, exist_ok=True)

    # Compute overlapping
    info_redu, info_gain = {}, {}
    for img_name_0, img_name_1 in itertools.combinations(poses.keys(), 2):
        print(f"{img_name_0} - {img_name_1}")

        img_path_0 = os.path.join(original_scene_path, img_name_0)
        img_path_1 = os.path.join(original_scene_path, img_name_1)
        estimator(Path(scene_path), [img_path_0], img_path_1, None, None, None, dict())

        ratio_A2B = dict()
        cams_old = inv(estimator.scene.get_im_poses())
        K_old = estimator.scene.get_intrinsics()
        all_pts3d_old = estimator.scene.get_pts3d()

        # NOTE(gogojjh): ptcloud may be expressed in camera1 or camera2\
        # TODO(gogojjh): fix the issues of correct coordinates
        confs = []
        for i in range(len(all_pts3d_old)):
            conf = float(estimator.scene.conf_i[edge_str(i, 1-i)].mean() * estimator.scene.conf_j[edge_str(i, 1-i)].mean())
            confs.append(conf)

        if confs[1] > confs[0]:
            cams = [cams_old[1], cams_old[0]]
            K = [K_old[1], K_old[0]]
            all_pts3d = [all_pts3d_old[1], all_pts3d_old[0]]
        else:
            cams = [cams_old[0], cams_old[1]]
            K = [K_old[0], K_old[1]]
            all_pts3d = [all_pts3d_old[0], all_pts3d_old[1]]

        for i, pts3d in enumerate(all_pts3d):
            for j in range(len(all_pts3d)):
                if i == j: continue
                print(f"{i} -> {j}")
                print(cams[j])

                pcd = save_point_cloud(pts3d.detach().cpu().numpy(), os.path.join(output_dir, f'pts3d_{i}.pcd'), True)
                
                pts3d_flat = pts3d.reshape(-1, 3)
                ori_depth = pts3d_flat[:, 2]

                proj = geotrf(cams[j], pts3d_flat) # project the point in i into j
                proj_depth = proj[:, 2]
                u, v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

                H, W, _ = pts3d.shape
                msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
                msk = (proj_depth[msk_i] / ori_depth[msk_i]) < 1.2

                import matplotlib.pyplot as plt
                # Create figure
                fig, axs = plt.subplots(2, 2, figsize=(16, 12))

                # Original Depth Map
                im0 = axs[0,0].imshow(ori_depth.reshape(288, 512, 1).detach().cpu().numpy(), cmap='turbo')
                axs[0,0].set_title(f'Original Depth (Camera {i})')
                plt.colorbar(im0, ax=axs[0,0], label='Depth')

                # Projected Depth Map
                im1 = axs[0,1].imshow(proj_depth.reshape(288, 512, 1).detach().cpu().numpy(), cmap='turbo')
                axs[0,1].set_title(f'Projected Depth (Camera {j})')
                plt.colorbar(im1, ax=axs[0,1], label='Depth')

                # Save and display
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'depth_maps_{i}_to_{j}.jpg'))
                plt.close()

                ratio_A2B[(i, j)] = np.sum(msk.detach().cpu().numpy()) / (H * W)

        info_redu[(img_name_0, img_name_1)] = ratio_A2B[(0, 1)]          # how much information is redundant of img_0
        info_gain[(img_name_0, img_name_1)] = 1.0 - ratio_A2B[(0, 1)]    # how much information is gained of img_0 
        info_redu[(img_name_1, img_name_0)] = ratio_A2B[(1, 0)]          # how much information is redundant of img_1
        info_gain[(img_name_1, img_name_0)] = 1.0 - ratio_A2B[(1, 0)]    # how much information is gained of img_1

        print(f'Info Redu: larger, the more of A is observed by B')
        print(f'(A to B) Info Redu: {ratio_A2B[(0, 1)]:.3f}, Info Gain: {1.0-ratio_A2B[(0, 1)]:.3f}')        
        print(f'(B to A) Info Redu: {ratio_A2B[(1, 0)]:.3f}, Info Gain: {1.0-ratio_A2B[(1, 0)]:.3f}')

        input()

    np.save(os.path.join(scene_path, 'information_redundancy.npy'), info_redu)
    np.save(os.path.join(scene_path, 'information_gain.npy'), info_gain)

    # Copy timestamps
    import shutil
    timestamp_file = os.path.join(original_scene_path, 'timestamps.txt')
    destination_file = os.path.join(scene_path, 'timestamps.txt')
    shutil.copy2(timestamp_file, destination_file)
    print(f"Copied {timestamp_file} to {destination_file}")    

def load_scene_data(dataset_path, scene):
    """Improved data loading with submap structure conversion"""
    scene_path = os.path.join(dataset_path, scene)
    required_files = ['iqa.txt', 'information_redundancy.npy', 'information_gain.npy', 'submap_split.npy', 'timestamps.txt', 'descriptors.txt']
    
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
        'timestamps': read_timestamps(os.path.join(scene_path, 'timestamps.txt')),
        'descriptors': read_descriptors(os.path.join(scene_path, 'descriptors.txt')),
        'iqa_scores': read_timestamps(os.path.join(scene_path, 'iqa.txt')),
        'info_redu': np.load(os.path.join(scene_path, 'information_redundancy.npy'), allow_pickle=True),
        'info_gain': np.load(os.path.join(scene_path, 'information_gain.npy'), allow_pickle=True),
        'submap_splits': submap_splits,
    }

def select_keyframes(scene_data, args):
    ###### Definition
    ###### timestamps[img_name] = timestamp
    ###### iqa_scores[img_name] = iqa_score 
    ###### info_redu[img_name0, img_name1] = info_redu
    ###### info_gain[img_name0, img_name1] = info_gain
    ###### submap_splits[i] = {'start': start_time, 'end': end_time, 'frames': [img_name0, img_name1, ...]}
    timestamps = scene_data['timestamps']       
    descriptors = scene_data['descriptors']
    iqa_scores = scene_data['iqa_scores']       
    info_redu = scene_data['info_redu'].item()  
    info_gain = scene_data['info_gain'].item()  
    submap_splits = scene_data['submap_splits']
    submap_database = submap_splits[:-1]
    
    if args.method == 'landmark':
        kf_selector = LandmarkSelector()

    keyframes = kf_selector.select_keyframes(timestamps, descriptors, iqa_scores, info_redu, info_gain, submap_database)

    return keyframes

def main():
    args = parse_arguments()
    scene_data = load_scene_data(args.dataset_path, args.scene)
    print(f"Loaded data with {len(scene_data['submap_splits'])} submaps")

    keyframes = select_keyframes(scene_data, args)

if __name__ == "__main__":
    main()
