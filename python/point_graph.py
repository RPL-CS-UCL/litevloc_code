import os
import numpy as np

from pathlib import Path

from point_node import PointNode
from utils.utils_shortest_path import dijk_shortest_path
from utils.base_graph import BaseGraph
from utils.utils_geom import read_timestamps, read_poses, convert_pose_inv

class PointGraphLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(map_root: Path, edge_type: str):
		point_graph = PointGraph(map_root, edge_type)
		
		# Read timestamps, poses, and poses_abs_gt
		times = read_timestamps(str(map_root/'timestamps.txt'))
		poses = read_poses(str(map_root/'poses.txt'))
		poses_abs_gt = read_poses(str(map_root/'poses_abs_gt.txt'))

		if poses is not None:
			for key in poses.keys():	
				time = times[key][0]
				# qw, qx, qy, tx, ty, tz
				trans, quat = convert_pose_inv(poses[key][4:], np.roll(poses[key][:4], -1), 'xyzw')
				
				node_id = point_graph.get_num_node()
				node = PointNode(node_id, f'point node {node_id}', time, trans, quat, None, None)
				node.set_pose(trans, quat)
				if poses_abs_gt is not None and key in poses_abs_gt:
					trans, quat = convert_pose_inv(
						poses_abs_gt[key][4:], 
						np.roll(poses_abs_gt[key][:4], -1), 
						'xyzw'
					)
					node.set_pose_gt(trans, quat)

				point_graph.add_node(node)

		edge_list_path = map_root/f"edges_{edge_type}.txt"
		point_graph.read_edge_list(edge_list_path)

		return point_graph

# Image Graph Class
class PointGraph(BaseGraph):
	def __init__(self, map_root: Path, edge_type: str):
		super().__init__(map_root, edge_type)

	def save_to_file(self):
		edge_list_path = self.map_root/f"edges_{self.edge_type}.txt"
		self.write_edge_list(edge_list_path)

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