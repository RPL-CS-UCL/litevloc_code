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

	def __str__(self):
		num_edge = 0
		for node_id, node in self.nodes.items():
			num_edge += len(node.edges)
		out_str = f"Graph has {len(self.nodes)} nodes with {num_edge} edges, load data from {self.map_root}"
		return out_str

	@staticmethod
	def load_data(map_root, resize, depth_scale, 
		load_rgb=False, load_depth=False, normalized=False,
		edge_type='odometry'):
		"""
		Load data from the specified map directory and create an image graph.

		Args:
			map_root (str): The root directory of the map.
			resize (tuple): The desired size to resize the images to.
			depth_scale (float): The scale factor to apply to the depth images.
			load_rgb (bool, optional): Whether to load RGB images. Defaults to False.
			load_depth (bool, optional): Whether to load depth images. Defaults to False.
			normalized (bool, optional): Whether to normalize the RGB images. Defaults to False.
			edge_type (str, optional): The type of edges to read from the map directory. 
										Can be 'odometry', 'covisible', or 'traversable'. 
										Defaults to 'odometry'.

		Returns:
			ImageGraph: The created image graph.

		Raises:
			FileNotFoundError: If any of the required files are not found in the map directory.
		"""
		image_graph = ImageGraph(map_root)

		# Read timestamps, intrinsics, poses, poses_abs_gt, and descriptors
		timestamps = read_timestamps(os.path.join(map_root, 'timestamps.txt'))
		intrinsics = read_intrinsics(os.path.join(map_root, 'intrinsics.txt'))
		poses = read_poses(os.path.join(map_root, 'poses.txt'))
		poses_abs_gt = read_poses(os.path.join(map_root, 'poses_abs_gt.txt'))
		descs = read_descriptors(os.path.join(map_root, 'database_descriptors.txt'))

		# Iterate over each image and create observation nodes
		for key in poses.keys():
			rgb_img_name = key
			rgb_img_path = os.path.join(map_root, rgb_img_name)
			if not load_rgb:
				rgb_image = None
			elif load_rgb and os.path.exists(rgb_img_path):
				rgb_image = load_rgb_image(rgb_img_path, resize, normalized=normalized)
			else:
				continue

			depth_img_name = key.replace('color.jpg', 'depth.png')
			depth_img_path = os.path.join(map_root, depth_img_name)
			if not load_depth:
				depth_image = None
			elif load_depth and os.path.exists(depth_img_path):
				depth_image = load_depth_image(os.path.join(map_root, depth_img_name), depth_scale=depth_scale)
			else:
				continue

			# Extract extrinsics
			time, quat, trans = timestamps[key], poses[key][:4], poses[key][4:]
			Tc2w = convert_vec_to_matrix(trans, quat, 'wxyz')
			trans, quat = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')

			# Extract intrinsics
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

			# Extract descriptors
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

		# Read edge list based on the specified edge type
		if edge_type == 'odometry':
			edge_list_path = os.path.join(map_root, 'odometry_edge_list.txt')
		elif edge_type == 'covisible':
			edge_list_path = os.path.join(map_root, 'covisible_edge_list.txt')
		elif edge_type == 'traversable':
			edge_list_path = os.path.join(map_root, 'traversable_edge_list.txt')
		image_graph.read_edge_list(edge_list_path)

		return image_graph

# Image Graph Class
class ImageGraph(BaseGraph):
	def __init__(self, map_root):
		super().__init__()
		self.map_root = map_root
		
	def save_to_file(self):
		num_node = self.get_num_node()
		times = np.empty((num_node, 2), dtype=object)
		intrinsics = np.empty((num_node, 7), dtype=object)
		poses = np.empty((num_node, 8), dtype=object)
		poses_abs_gt = np.empty((num_node, 8), dtype=object)
		first_node = next(iter(self.nodes.values()))
		descs = np.empty((num_node, len(first_node.get_descriptor()) + 1), dtype=object)
		for line_id, (node_id, node) in enumerate(self.nodes.items()):
			img_name = f"seq/{node_id:06d}.color.jpg"
			times[line_id, 0], times[line_id, 1] = img_name, node.time

			fx, fy, cx, cy, width, height = \
				node.raw_K[0, 0], node.raw_K[1, 1], node.raw_K[0, 2], node.raw_K[1, 2], \
				node.raw_img_size[0], node.raw_img_size[1]
			intrinsics[line_id, 0], intrinsics[line_id, 1:] = img_name, np.array([fx, fy, cx, cy, width, height])
			
			Tw2c = convert_vec_to_matrix(node.trans, node.quat, 'xyzw')
			Tc2w = np.linalg.inv(Tw2c)
			trans, quat = convert_matrix_to_vec(Tc2w, 'wxyz')
			poses[line_id, 0], poses[line_id, 1:5], poses[line_id, 5:] = img_name, quat, trans

			Tw2c = convert_vec_to_matrix(node.trans_gt, node.quat_gt, 'xyzw')
			Tc2w = np.linalg.inv(Tw2c)
			trans, quat = convert_matrix_to_vec(Tc2w, 'wxyz')
			poses_abs_gt[line_id, 0], poses_abs_gt[line_id, 1:5], poses_abs_gt[line_id, 5:] = img_name, quat, trans

			descs[line_id, 0], descs[line_id, 1:] = img_name, node.get_descriptor()

		np.savetxt(os.path.join(self.map_root, "timestamps.txt"), times, fmt='%s %.6f')
		np.savetxt(os.path.join(self.map_root, "intrinsics.txt"), intrinsics, fmt='%s %.6f %.6f %.6f %.6f %d %d')
		# NOTE(gogojjh): poses.txt is not updated and used
		np.savetxt(os.path.join(self.map_root, "poses_abs_gt.txt"), poses_abs_gt, fmt='%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f')
		np.savetxt(os.path.join(self.map_root, "database_descriptors.txt"), descs, fmt='%s ' + '%.6f ' * (descs.shape[1] - 1))
		self.write_edge_list(os.path.join(self.map_root, 'odometry_edge_list.txt'))

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