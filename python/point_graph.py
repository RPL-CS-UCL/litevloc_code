import os
import numpy as np

from point_node import PointNode

from pycpptools.src.python.utils_algorithm.shortest_path import dijk_shortest_path
from pycpptools.src.python.utils_algorithm.base_graph import BaseGraph
from pycpptools.src.python.utils_math.tools_eigen import convert_vec_to_matrix, convert_matrix_to_vec

def read_timestamps(file_path):
	times = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				data = float(line.strip().split(' ')[1]) # Each row: image_name, timestamp
			else:
				data = float(line.strip().split(' ')[1]) # Each row: qw, qx, qy, tx, ty, tz
			times[line_id] = np.array(data)
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
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, qw, qx, qy, tx, ty, tz
			else:
				data = [float(p) for p in line.strip().split(' ')] # Each row: qw, qx, qy, tx, ty, tz
			poses[line_id] = np.array(data)
	return poses

class PointGraphLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(graph_path):
		point_graph = PointGraph()
		times = read_timestamps(os.path.join(graph_path, 'timestamps.txt'))
		poses = read_poses(os.path.join(graph_path, 'poses.txt'))

		for key in poses.keys():	
			# Each row: time, qw, qx, qy, tx, ty, tz
			time, quat, trans = times[key], poses[key][:4], poses[key][4:] 
			Tc2w = convert_vec_to_matrix(trans, quat, 'wxyz')
			trans, quat = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
			
			node_id = point_graph.get_num_node()
			node = PointNode(node_id, f'point node {node_id}', time, trans, quat, None, None)
			point_graph.add_node(node)

		edge_file = os.path.join(graph_path, 'edge_list.txt')
		point_graph.read_edge_list(edge_file)
		
		return point_graph

# Image Graph Class
class PointGraph(BaseGraph):
	def __init__(self):
		super().__init__()

class TestPointGraph():
	def __init__(self):
		pass
	
	def run_test(self):
		# Initialize the point graph
		graph = PointGraph()

		# Add nodes to the graph
		graph.add_node(PointNode(1, "descriptor_1", 
								0, np.zeros((1, 3)), np.zeros((1, 4))))
		graph.add_node(PointNode(2, "descriptor_2", 
								0, np.zeros((1, 3)), np.zeros((1, 4))))
		graph.add_node(PointNode(3, "descriptor_3", 
								0, np.zeros((1, 3)), np.zeros((1, 4))))
		graph.add_node(PointNode(4, "descriptor_4", 
								0, np.zeros((1, 3)), np.zeros((1, 4))))

		# Add edges between the nodes with weights
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(2), 1.0)
		graph.add_edge_undirected(graph.get_node(2), graph.get_node(3), 2.0)
		graph.add_edge_undirected(graph.get_node(3), graph.get_node(4), 4.0)
		graph.add_edge_undirected(graph.get_node(1), graph.get_node(4), 4.0)

		# Get the image descriptor of a specific node
		node = graph.get_node(2)
		print(f"Image Descriptor of Node 2: {node.global_descriptor}")

		# Find the shortest path from node 1 to node 4
		distance, path = dijk_shortest_path(graph, graph.get_node(1), graph.get_node(4))
		print(f"Shortest Path from Node 1 to Node 4 with distance {distance}")
		print(' -> '.join([str(node.id) for node in path]))

if __name__ == '__main__':
	test_point_graph = TestPointGraph()
	test_point_graph.run_test()