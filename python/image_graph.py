import os
import numpy as np

import heapq

from utils.utils_image import load_rgb_image, load_depth_image

from pycpptools.src.python.utils_algorithm.shortest_path import dijkstra_shortest_path
from pycpptools.src.python.utils_algorithm.base_node import BaseNode
from pycpptools.src.python.utils_algorithm.base_graph import BaseGraph

class ImageGraphLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(image_graph_path, image_size, depth_scale, normalized=False, num_sample=1, num_load=10000):
		image_graph = ImageGraph()

		# Each row: time, tx, ty, tz, qx, qy, qz, qw
		poses_w_cam = np.loadtxt(os.path.join(image_graph_path, 'camera_pose_gt.txt'))
		for i in range(0, 
								 	 min(poses_w_cam.shape[0], num_load * num_sample), 
									 num_sample):
			rgb_img_path = os.path.join(image_graph_path, 'rgb', f'{i:06}.png')
			rgb_image = load_rgb_image(rgb_img_path, image_size, normalized=normalized)

			depth_img_path = os.path.join(image_graph_path, 'depth', f'{i:06}.png')
			depth_image = load_depth_image(depth_img_path, image_size, depth_scale=depth_scale)
			
			time, t_w_cam, quat_w_cam = poses_w_cam[i, 0], poses_w_cam[i, 1:4], poses_w_cam[i, 4:] 

			node = ImageNode(i, 
							         rgb_image, depth_image, f'camera node {i}', 
							         time, t_w_cam, quat_w_cam, 
							         rgb_img_path, depth_img_path)
			image_graph.add_node(node)

			if i / num_sample > num_load:
				break

		return image_graph

# Image Node Class	
class ImageNode(BaseNode):
	def __init__(self, id, 
							 rgb_image, depth_image, global_descriptor, 
							 time, t_w_cam, quat_w_cam, 
							 rgb_img_path, depth_img_path):
		super().__init__(id)

		self.rgb_image = rgb_image
		self.depth_image = depth_image

		self.global_descriptor = global_descriptor

		self.time = time
		self.t_w_cam = t_w_cam
		self.quat_w_cam = quat_w_cam

		self.rgb_img_path = rgb_img_path
		self.depth_img_path = depth_img_path

	def set_descriptor(self, global_descriptor):
		self.global_descriptor = global_descriptor

	def get_descriptor(self):
		return self.global_descriptor

# Image Graph Class
class ImageGraph(BaseGraph):
	def __init__(self):
		super().__init__()

class TestImageGraph():
	def __init__(self):
		pass
	
	def run_test(self):
		# Initialize the image graph
		graph = ImageGraph()

		# Add nodes to the graph
		graph.add_node(ImageNode(1, None, None, "descriptor_1", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									'tmp_rgb.png', 'tmp_depth.png'))
		graph.add_node(ImageNode(2, None, None, "descriptor_2", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									'tmp_rgb.png', 'tmp_depth.png'))
		graph.add_node(ImageNode(3, None, None, "descriptor_3", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									'tmp_rgb.png', 'tmp_depth.png'))
		graph.add_node(ImageNode(4, None, None, "descriptor_4", 
									0, np.zeros((1, 3)), np.zeros((1, 4)), 
									'tmp_rgb.png', 'tmp_depth.png'))

		# Add edges between the nodes with weights
		graph.add_edge(graph.get_node(1), graph.get_node(2), 1.0)
		graph.add_edge(graph.get_node(2), graph.get_node(3), 2.0)
		graph.add_edge(graph.get_node(3), graph.get_node(4), 1.0)
		graph.add_edge(graph.get_node(1), graph.get_node(4), 4.0)

		# Get the image descriptor of a specific node
		node = graph.get_node(2)
		print(f"Image Descriptor of Node 2: {node.global_descriptor}")

		# Find the shortest path from node 1 to node 4
		distance, path = dijkstra_shortest_path(graph, graph.get_node(1), graph.get_node(4))
		print(f"Shortest Path from Node 1 to Node 4: {path} with distance {distance}")

		# Find all connections of a specific node
		connections = graph.find_connection(2)
		print(f"Connections of Node 2: {connections}")

		# Verify the new connection
		connections = graph.find_connection(2)
		print(f"Connections of Node 2 after adding a new connection: {connections}")

		# Find the shortest path again from node 1 to node 4 after adding new connection
		distance, path = dijkstra_shortest_path(graph, graph.get_node(1), graph.get_node(4))
		print(f"New Shortest Path from Node 1 to Node 4: {path} with distance {distance}")

if __name__ == '__main__':
	test_image_graph = TestImageGraph()
	test_image_graph.run_test()