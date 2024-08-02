# [Computation]:
# python submission.py --config ../config/dataset/mapfree.yaml --models master --pose_solver essentialmatrixmetric --out_dir /Titan/dataset/data_mapfree/results --split val
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetric/submission.zip --split val --log error

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import time
import torch
import numpy as np
from tqdm import tqdm

from transforms3d.quaternions import mat2quat

from pose_solver import available_solvers, get_solver
from matching import available_models, get_matcher
from matching.utils import to_numpy, get_image_pairs_paths

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from utils.utils_image_matching_method import save_visualization

import pycpptools.src.python.utils_math as pytool_math

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map-free-reloc"))
from config.default import cfg

def depth_image_to_point_cloud(depth_image, intrinsics, image_shape):
    """
    Convert a depth image to a point cloud.

    Parameters:
    depth_image (numpy.ndarray): The depth image.
    intrinsics (numpy.ndarray): The camera intrinsic matrix.

    Returns:
    numpy.ndarray: The point cloud as an (N, 3) array.
    """
    w, h = image_shape
    i, j = np.indices((h, w))
    z = depth_image
    x = (j - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (i - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def transform_point_cloud(points, transformation_matrix):
    """
    Apply a transformation to a point cloud.

    Parameters:
    points (numpy.ndarray): The point cloud as an (N, 3) array.
    transformation_matrix (numpy.ndarray): The 4x4 transformation matrix.

    Returns:
    numpy.ndarray: The transformed point cloud as an (N, 3) array.
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = points_homogeneous @ transformation_matrix.T
    return points_transformed[:, :3]

def project_point_cloud(points, intrinsics, image_shape):
    """
    Project a point cloud onto an image plane.

    Parameters:
    points (numpy.ndarray): The point cloud as an (N, 3) array.
    intrinsics (numpy.ndarray): The camera intrinsic matrix.
    image_shape (tuple): The shape of the output image (height, width).

    Returns:
    numpy.ndarray: The projected depth image.
    """
    w, h = image_shape
    z = points[:, 2]
    x = (points[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]).astype(np.int32)
    y = (points[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]).astype(np.int32)

    depth_image = np.zeros((h, w))
    valid_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z > 0)
    depth_image[y[valid_mask], x[valid_mask]] = z[valid_mask]
    return depth_image	

def eval(args):
    K = np.array([542.7908325195312, 0.0, 481.3011474609375, 0.0, 542.7908325195312, 271.8500671386719, 0.0, 0.0, 1.0]).reshape(3, 3)
    width, height = 960, 540

    quat = np.array([-0.002553716, 0.023818718, -0.008624582, 0.999675829]) 
    trans = np.array([-0.315558359, 0.430753200, -1.943453622])
    T10 = pytool_math.tools_eigen.convert_vec_to_matrix(trans, quat, 'xyzw')

    depth_img0 = np.random.rand(height, width) * 10.0
    depth_points = depth_image_to_point_cloud(depth_img0, K, (width, height))
    depth_points_transformed = transform_point_cloud(depth_points, T10)
    depth_img1 = project_point_cloud(depth_points_transformed, K, (width, height))

    mkpts0 = np.random.rand(10, 2) * 640  # Random 2D points

    # Load configs
    cfg.merge_from_file(args.config)
    for pose_solver in available_solvers:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    eval(args)
