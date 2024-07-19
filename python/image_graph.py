import os
import numpy as np

from utils.utils_image import load_rgb_image, load_depth_image
from image_node import ImageNode

from pycpptools.src.python.utils_algorithm.shortest_path import dijk_shortest_path
from pycpptools.src.python.utils_algorithm.base_graph import BaseGraph

class ImageGraphLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(graph_path, image_size, depth_scale, normalized=False):
		image_graph = ImageGraph()
		poses = np.loadtxt(os.path.join(graph_path, 'camera_pose_gt.txt'))
		descs_path = os.path.join(graph_path, 'database_descriptors.npy')
		if os.path.exists(descs_path):
			descs = np.load(descs_path)
		else:	
			descs = None

		for i in range(0, poses.shape[0]):
			rgb_img_path = os.path.join(graph_path, 'rgb', f'{i:06}.png')
			rgb_image = load_rgb_image(rgb_img_path, image_size, normalized=normalized)
			depth_img_path = os.path.join(graph_path, 'depth', f'{i:06}.png')
			depth_image = load_depth_image(depth_img_path, image_size, depth_scale=depth_scale)

			# Each row: time, tx, ty, tz, qx, qy, qz, qw			
			time, trans, quat = poses[i, 0], poses[i, 1:4], poses[i, 4:] 
			node = ImageNode(i, 
											 rgb_image, depth_image, f'image node {i}', 
											 time, trans, quat, 
											 rgb_img_path, depth_img_path)
			node.set_pose_gt(trans, quat)
			if descs is not None:
				node.set_descriptor(descs[i, :])

			image_graph.add_node(node)
		image_graph.read_edge_list(os.path.join(graph_path, 'edge_list.txt'))
		return image_graph

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
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(2), 1.0)
		graph.add_edge_undirected(graph.get_node(2), graph.get_node(3), 2.0)
		graph.add_edge_undirected(graph.get_node(3), graph.get_node(4), 1.0)
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(4), 4.0)

		# Get the image descriptor of a specific node
		node = graph.get_node(2)
		print(f"Image Descriptor of Node 2: {node.global_descriptor}")

		# Find the shortest path from node 1 to node 4
		distance, path = dijk_shortest_path(graph, graph.get_node(1), graph.get_node(4))
		print(f"Shortest Path from Node 1 to Node 4 with distance {distance}")
		print(' -> '.join([str(node.id) for node in path]))
		
if __name__ == '__main__':
	test_image_graph = TestImageGraph()
	test_image_graph.run_test()