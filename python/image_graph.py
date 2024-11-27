#! /usr/bin/env python

import os
import numpy as np
import pathlib

from utils.utils_image import load_rgb_image, load_depth_image
from image_node import ImageNode

from pycpptools.src.python.utils_algorithm.base_graph import BaseGraph
from pycpptools.src.python.utils_sensor.utils import correct_intrinsic_scale
from pycpptools.src.python.utils_math.tools_eigen import convert_vec_to_matrix, convert_matrix_to_vec

def read_timestamps(file_path):
	times = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = float(line.strip().split(' ')[1]) # Each row: image_name, timestamp
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				data = float(line.strip().split(' ')[1]) # Each row: qw, qx, qy, tx, ty, tz
			times[img_name] = np.array(data)
	return times

def read_poses(file_path):
	if not os.path.exists(file_path):
		print(f"Poses not found in {file_path}")
		return None

	poses = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, qw, qx, qy, tx, ty, tz
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				data = [float(p) for p in line.strip().split(' ')] # Each row: qw, qx, qy, tx, ty, tz
			poses[img_name] = np.array(data)
	return poses

def read_intrinsics(file_path):
	if not os.path.exists(file_path):
		print(f"Intrinsics not found in {file_path}")
		return None

	intrinsics = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, fx fy cx cy width height
			else:
				img_name = f'{line_id:06}.color.jpg'
				data = [float(p) for p in line.strip().split(' ')] # Each row: fx fy cx cy width height
			intrinsics[img_name] = np.array(data)
	return intrinsics

def read_descriptors(file_path):
	if not os.path.exists(file_path):
		print(f"Descriptors not found in {file_path}")
		return None
	
	descs = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, descriptor (a vector)
				descs[img_name] = np.array(data)
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				descs[img_name] = np.array([float(p) for p in line.strip().split(' ')])
	return descs		

class ImageGraphLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(map_root, resize, depth_scale, load_rgb=False, load_depth=False, normalized=False):
		image_graph = ImageGraph()
		image_graph.map_root = map_root
	
		timestamps = read_timestamps(os.path.join(map_root, 'timestamps.txt'))		
		intrinsics = read_intrinsics(os.path.join(map_root, 'intrinsics.txt'))
		poses = read_poses(os.path.join(map_root, 'poses.txt'))
		poses_abs_gt = read_poses(os.path.join(map_root, 'poses_abs_gt.txt'))
		descs = read_descriptors(os.path.join(map_root, 'database_descriptors.txt'))

		# NOTE(gogojjh): guarantee that each image has a corresponding pose
		for key in poses.keys():
			rgb_img_name = key
			rgb_img_path = os.path.join(map_root, rgb_img_name)
			if os.path.exists(rgb_img_path):
				rgb_image = load_rgb_image(rgb_img_path, resize, normalized=normalized) if load_rgb else None
			else:
				continue

			depth_img_name = key.replace('color.jpg', 'depth.png')
			depth_img_path = os.path.join(map_root, depth_img_name)
			if os.path.exists(depth_img_path):
				depth_image = load_depth_image(os.path.join(map_root, depth_img_name), depth_scale=depth_scale) if load_depth else None
			else:
				continue

			# Extrinsics
			time, quat, trans = timestamps[key], poses[key][:4], poses[key][4:]
			Tc2w = convert_vec_to_matrix(trans, quat, 'wxyz')
			trans, quat = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')

			# Intrinsics
			if key in intrinsics:
				fx, fy, cx, cy, width, height = \
					intrinsics[key][0], intrinsics[key][1], intrinsics[key][2], \
					intrinsics[key][3], int(intrinsics[key][4]), int(intrinsics[key][5])
			else:
				continue
			raw_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
			raw_img_size = np.array([width, height])
			K = correct_intrinsic_scale(raw_K, resize[0] / raw_img_size[0], resize[1] / raw_img_size[1]) if resize is not None else raw_K
			img_size = np.array([int(resize[0]), int(resize[1])]) if resize is not None else raw_img_size

			# Descriptors
			if descs is not None:
				if key in descs:
					desc = descs[key]
				else:
					continue
			else:
				desc = None

			# Create observation node
			node_id = image_graph.get_num_node()
			node = ImageNode(node_id, rgb_image, depth_image, desc,
							 time, trans, quat, 
							 K, img_size,
							 rgb_img_name, depth_img_name)
			node.set_raw_intrinsics(raw_K, raw_img_size)
			node.set_pose(trans, quat)
			if poses_abs_gt is not None and key in poses_abs_gt:
				quat, trans = poses_abs_gt[key][:4], poses_abs_gt[key][4:]
				Tc2w = convert_vec_to_matrix(trans, quat, 'wxyz')
				trans, quat = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
				node.set_pose_gt(trans, quat)
			image_graph.add_node(node)

		edge_list_path = os.path.join(map_root, 'edge_list.txt')
		image_graph.read_edge_list(edge_list_path)

		return image_graph

# Image Graph Class
class ImageGraph(BaseGraph):
	def __init__(self):
		super().__init__()

	# TODO(gogojjh):
	# def save_to_file(self):
	# 	os.makedirs(os.path.join(self.map_root, "seq"), exist_ok=True)
		

class TestImageGraph():
	def __init__(self):
		pass
	
	def run_test(self):
		# Initialize the image graph
		graph = ImageGraph()

		# Add nodes to the graph
		graph.add_node(ImageNode(1, None, None, "descriptor_1", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									np.eye(3), (640, 480),
									'tmp_rgb.png', 'tmp_depth.png'))
		graph.add_node(ImageNode(2, None, None, "descriptor_2", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									np.eye(3), (640, 480),
									'tmp_rgb.png', 'tmp_depth.png'))
		graph.add_node(ImageNode(3, None, None, "descriptor_3", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									np.eye(3), (640, 480),
									'tmp_rgb.png', 'tmp_depth.png'))
		graph.add_node(ImageNode(4, None, None, "descriptor_4", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									np.eye(3), (640, 480),
									'tmp_rgb.png', 'tmp_depth.png'))

		# Add edges between the nodes with weights
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(2), 1.0)
		graph.add_edge_undirected(graph.get_node(2), graph.get_node(3), 2.0)
		graph.add_edge_undirected(graph.get_node(3), graph.get_node(4), 1.0)
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(4), 4.0)

		# Get the image descriptor of a specific node
		node = graph.get_node(2)
		print(f"Image Descriptor of Node 2: {node.global_descriptor}")

		# Find the shortest path from node 1 to node 4
		from pycpptools.src.python.utils_algorithm.shortest_path import dijk_shortest_path
		distance, path = dijk_shortest_path(graph, graph.get_node(1), graph.get_node(4))
		print(f"Shortest Path from Node 1 to Node 4 with distance {distance}")
		print(' -> '.join([str(node.id) for node in path]))
		
if __name__ == '__main__':
	test_image_graph = TestImageGraph()
	test_image_graph.run_test()