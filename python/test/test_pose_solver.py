# [Computation]:
# python submission.py --config ../config/dataset/mapfree.yaml --models master --pose_solver essentialmatrixmetric --out_dir /Titan/dataset/data_mapfree/results --split val
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetric/submission.zip --split val --log error

import os
import sys
import argparse
import unittest

import time
import numpy as np
import cv2

import pycpptools.src.python.utils_math as pytool_math

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../benchmark_loc"))
from pose_solver import available_solvers, get_solver

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

class TestPoseSolver():
	def run(self):
			config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/dataset/ucl_campus.yaml")
			cfg.merge_from_file(config_file)

			# Define the inputs
			width, height = 960, 540
			K = np.array([542.7908325195312, 0.0, 481.3011474609375, 0.0, 542.7908325195312, 271.8500671386719, 0.0, 0.0, 1.0]).reshape(3, 3)
			quat = np.array([-0.002553716, 0.223818718, -0.008624582, 0.99675829]) 
			trans = np.array([-1.315558359, 0.430753200, -1.943453622])
			T10 = pytool_math.tools_eigen.convert_vec_to_matrix(trans, quat, 'xyzw')

			depth_img0 = np.random.rand(height, width) * 10.0
			depth_points = depth_image_to_point_cloud(depth_img0, K, (width, height))
			depth_points_transformed = transform_point_cloud(depth_points, T10)
			depth_img1 = project_point_cloud(depth_points_transformed, K, (width, height))

			# Generate random 2D points
			mkpts0 = (np.random.rand(2000, 2) * np.array([width, height])).astype(np.int32)
			depth = depth_img0[mkpts0[:, 1], mkpts0[:, 0]]
			x = (mkpts0[:, 0] - K[0, 2]) * depth / K[0, 0]
			y = (mkpts0[:, 1] - K[1, 2]) * depth / K[1, 1]
			depth_points = np.stack((x, y, depth), axis=-1).reshape(-1, 3)
			depth_points_transformed = transform_point_cloud(depth_points, T10)
			z = depth_points_transformed[:, 2]
			x = depth_points_transformed[:, 0] * K[0, 0] / z + K[0, 2]
			y = depth_points_transformed[:, 1] * K[1, 1] / z + K[1, 2]
			valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > 0)
			mkpts1 = np.stack((x, y), axis=-1)[valid_mask].astype(np.int32)
			mkpts0 = mkpts0[valid_mask]
			print(mkpts0.shape, mkpts1.shape)

			depth_img0_noisy = depth_img0 + (np.random.randn(height, width) - 0.5) * 0.0
			depth_img1_noisy = depth_img1 + (np.random.randn(height, width) - 0.5) * 0.0
			mkpts0_noisy = mkpts0 + (np.random.randn(mkpts0.shape[0], 2) - 0.5) * 0.0
			mkpts1_noisy = mkpts1 + (np.random.randn(mkpts1.shape[0], 2) - 0.5) * 0.0

			for str_solver in available_solvers:
				if str_solver == 'procrustes': continue
				pose_solver = get_solver(str_solver, cfg)
				print("Testing", pose_solver)
				R, t, inliers = pose_solver.estimate_pose(
					mkpts0_noisy, mkpts1_noisy, 
					K, K, 
					depth_img0_noisy, depth_img1_noisy)
				T10_est = np.eye(4)
				T10_est[:3, :3] = R
				T10_est[:3, 3] = t.reshape(3,)
				trans_dis, rot_dis = pytool_math.tools_eigen.compute_relative_dis_TF(T10, T10_est)
				print("T10\n", T10)
				print("T10_est\n", T10_est)
				print('inliers: ', inliers)
				print('Trans and Rot error: ', trans_dis, rot_dis)
				print('')

if __name__ == "__main__":
	test_pose_solver = TestPoseSolver()
	test_pose_solver.run()
