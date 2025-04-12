#! /usr/bin/env python

import os
import numpy as np
import shutil

from pathlib import Path
from utils.utils_geom import read_timestamps, read_intrinsics, read_poses, read_descriptors, read_gps
from utils.utils_geom import convert_pose_inv, correct_intrinsic_scale
from utils.utils_image import load_rgb_image, load_depth_image
from utils.base_graph import BaseGraph
from image_node import ImageNode

class ImageGraphLoader:
	def __init__(self):
		pass

	def __str__(self):
		num_edge = 0
		for _, node in self.nodes.items():
			num_edge += len(node.edges)
		out_str = f"Graph has {len(self.nodes)} nodes with {num_edge} edges, load data from {self.map_root}"
		return out_str

	@staticmethod
	def load_data(
		map_root: Path, 
		resize: tuple, # WxH
		depth_scale: float, 
		load_rgb: bool = False, 
		load_depth: bool = False, 
		normalized: bool = False,
		edge_type: str = 'odometry'
	):
		"""
		Load data from the specified map directory and create an image graph.

		Args:
			map_root (Path): The root directory of the map.
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
		image_graph = ImageGraph(map_root, edge_type)

		# Read timestamps, intrinsics, poses, poses_abs_gt, and descriptors
		timestamps = read_timestamps(str(map_root/'timestamps.txt'))
		intrinsics = read_intrinsics(str(map_root/'intrinsics.txt'))
		poses = read_poses(str(map_root/'poses.txt')) # img_name, qw, qx, qy, tx, ty, tz
		poses_abs_gt = read_poses(str(map_root/'poses_abs_gt.txt'))
		descs = read_descriptors(str(map_root/'database_descriptors.txt'))
		gps_datas = read_gps(str(map_root/'gps_data.txt'))
		iqa_datas = read_timestamps(str(map_root/'iqa_data.txt'))

		# Iterate over each image and create observation nodes
		if poses is not None:
			for node_id, key in enumerate(poses.keys()):
				# Read rgb image
				rgb_img_name = key
				rgb_img_path = os.path.join(str(map_root/key))
				if not load_rgb:
					rgb_image = None
				elif load_rgb and os.path.exists(rgb_img_path):
					rgb_image = load_rgb_image(rgb_img_path, resize, normalized=normalized)
				else:
					continue

				# Read depth image
				depth_img_name = key.replace('color.jpg', 'depth.png')
				depth_img_path = str(map_root/depth_img_name)
				if not load_depth:
					depth_image = None
				elif load_depth and os.path.exists(depth_img_path):
					depth_image = load_depth_image(depth_img_path, depth_scale=depth_scale)
				else:
					continue

				time = timestamps[key][0]

				# Extract extrinsics
				trans, quat = convert_pose_inv(
					poses[key][4:], 
					np.roll(poses[key][:4], -1), 
					'xyzw'
				)

				# Extract intrinsics
				if key in intrinsics:
					fx, fy, cx, cy, width, height = \
						intrinsics[key][0], intrinsics[key][1], intrinsics[key][2], \
						intrinsics[key][3], int(intrinsics[key][4]), int(intrinsics[key][5])
				else:
					continue
				raw_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
				raw_img_size = np.array([int(width), int(height)])
				if resize is not None:
					K = correct_intrinsic_scale(raw_K, resize[0] / width, resize[1] / height) 
					img_size = np.array([int(resize[0]), int(resize[1])])
				else:
					K = raw_K
					img_size = raw_img_size

				# Extract descriptors
				if descs is not None:
					if key in descs:
						desc = descs[key]
					else:
						continue
				else:
					desc = None

				# Extract GPS
				if gps_datas is not None:
					if key in gps_datas:
						gps_data = gps_datas[key]
					else:
						continue
				else:
					gps_data = None

				# Extract Image Quality Assessment data
				if iqa_datas is not None:
					if key in iqa_datas:
						iqa_data = iqa_datas[key][0]
					else:
						continue
				else:
					iqa_data = None

				# Create observation node
				node = ImageNode(
					node_id, rgb_image, depth_image, desc,
					time, trans, quat, 
					K, img_size,
					rgb_img_name, depth_img_name,
					gps_data, iqa_data
				)
				# Set other variables
				node.set_raw_intrinsics(raw_K, raw_img_size)
				node.set_pose(trans, quat)
				if poses_abs_gt is not None and key in poses_abs_gt:
					trans, quat = convert_pose_inv(
						poses_abs_gt[key][4:], 
						np.roll(poses_abs_gt[key][:4], -1), 
						'xyzw'
					)
					node.set_pose_gt(trans, quat)
			
				# Add the new node into the graph
				image_graph.add_node(node)

		# Read edge list based on the specified edge type
		edge_list_path = map_root/f"edges_{edge_type}.txt"
		image_graph.read_edge_list(edge_list_path)

		return image_graph

# Image Graph Class
class ImageGraph(BaseGraph):
	def __init__(self, map_root: Path, edge_type: str):
		super().__init__(map_root, edge_type)
		
	def copy_sensor_data(self, other_graph: 'ImageGraph'):
		"""Copy sensor data files from source map"""
		for node in other_graph.nodes.values():
			# Handle RGB images
			if node.rgb_img_name:
				src = other_graph.map_root / node.rgb_img_name
				if src.exists():
					new_name = f"seq/{node.id:06d}.color.jpg"
					dest = self.map_root / new_name
					shutil.copy(src, dest)
					node.rgb_img_name = new_name

			# Handle depth images
			if node.depth_img_name:
				src = other_graph.map_root / node.depth_img_name
				if src.exists():
					new_name = f"seq/{node.id:06d}.depth.png"
					dest = self.map_root / new_name
					shutil.copy(src, dest)
					node.depth_img_name = new_name

	def rm_sensor_data(
			self, 
			nodes_to_remove: list # [node0, node1, ...]
		):
		"""Remove sensor data files from source map if the corresponding node is removed"""
		for node in nodes_to_remove:
			if node.rgb_img_name:
				src = self.map_root / node.rgb_img_name
				if src.exists():
					src.unlink()
			
			if node.depth_img_name:
				src = self.map_root / node.depth_img_name
				if src.exists():
					src.unlink()
	def save_to_file(self, edge_only=False):
		if not edge_only:
			num_node = self.get_num_node()
			first_node = next(iter(self.nodes.values()))
			intrinsics = np.empty((num_node, 7), dtype=object)
			iqa_data = np.empty((num_node, 2), dtype=object)
			descs = np.empty((num_node, len(first_node.get_descriptor()) + 1), dtype=object)
			for line_id, (node_id, node) in enumerate(self.nodes.items()):
				# Force the image name to be concistent with the node id
				img_name = f"seq/{node_id:06d}.color.jpg"
				fx, fy, cx, cy, width, height = \
					node.raw_K[0, 0], node.raw_K[1, 1], node.raw_K[0, 2], node.raw_K[1, 2], \
					node.raw_img_size[0], node.raw_img_size[1]
				intrinsics[line_id, :] = [img_name, fx, fy, cx, cy, width, height]
				iqa_data[line_id, :] = [img_name, node.iqa_data]
				descs[line_id, :] = [img_name, *node.get_descriptor()]

			np.savetxt(str(self.map_root/"intrinsics.txt"), intrinsics, fmt='%s' + ' %.6f'*4 + ' %d %d')
			np.savetxt(str(self.map_root/"iqa_data.txt"), iqa_data, fmt='%s %.6f')
			np.savetxt(str(self.map_root/"database_descriptors.txt"), descs, fmt='%s' + ' %.6f'*(descs.shape[1] - 1))

		self.write_edge_list(self.map_root/f"edges_{self.edge_type}.txt")

class TestImageGraph():
	def __init__(self):
		pass
	
	def run_test(self):
		# Initialize the image graph
		graph = ImageGraph(map_root=Path('/tmp'))

		# Add nodes to the graph
		graph.add_node(ImageNode(
			1, None, None, "descriptor_1", 
			0, np.zeros((1, 3)), np.zeros((1, 4)), 
			np.eye(3), (640, 480),
			'tmp_rgb.png', 'tmp_depth.png')
		)
		graph.add_node(ImageNode(
			2, None, None, "descriptor_2", 
			0, np.zeros((1, 3)), np.zeros((1, 4)), 
			np.eye(3), (640, 480),
			'tmp_rgb.png', 'tmp_depth.png')
		)
		graph.add_node(ImageNode(
			3, None, None, "descriptor_3", 
			0, np.zeros((1, 3)), np.zeros((1, 4)), 
			np.eye(3), (640, 480),
			'tmp_rgb.png', 'tmp_depth.png')
		)
		graph.add_node(ImageNode(
			4, None, None, "descriptor_4", 
			0, np.zeros((1, 3)), np.zeros((1, 4)), 
			np.eye(3), (640, 480),
			'tmp_rgb.png', 'tmp_depth.png')
		)

		# Add edges between the nodes with weights
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(2), 1.0)
		graph.add_edge_undirected(graph.get_node(2), graph.get_node(3), 2.0)
		graph.add_edge_undirected(graph.get_node(3), graph.get_node(4), 1.0)
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(4), 4.0)

		# Get the image descriptor of a specific node
		node = graph.get_node(2)
		print(f"Image Descriptor of Node 2: {node.global_descriptor}")

		# Find the shortest path from node 1 to node 4
		from utils.utils_shortest_path import dijk_shortest_path
		distance, path = dijk_shortest_path(graph, graph.get_node(1), graph.get_node(4))
		print(f"Shortest Path from Node 1 to Node 4 with distance {distance}")
		print(' -> '.join([str(node.id) for node in path]))
		
if __name__ == '__main__':
	test_image_graph = TestImageGraph()
	test_image_graph.run_test()