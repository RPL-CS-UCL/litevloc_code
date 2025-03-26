import os
import numpy as np
from utils.gtsam_pose_graph import PoseGraph
from utils.utils_geom import convert_vec_gtsam_pose3

class BaseGraph:
	# Initialize an empty dictionary to store nodes
	def __init__(self):
		self.nodes = {}

	def __str__(self):
		num_edge = 0
		for node_id, node in self.nodes.items(): 
			num_edge += len(node.edges)
		out_str = f"Graph has {len(self.nodes)} nodes with {num_edge} edges"
		return out_str

	def read_edge_list(self, path_edge_list):
		if not os.path.exists(path_edge_list): 
			return
		edges_A_B_weight = np.loadtxt(path_edge_list, dtype=float)
		for edge in edges_A_B_weight:
			node_id0, node_id1 = int(edge[0]), int(edge[1])
			weight = edge[2]
			if (self.get_node(node_id0) is not None) and (self.get_node(node_id1) is not None):
				node0 = self.get_node(node_id0)
				node1 = self.get_node(node_id1)
				self.add_edge_undirected(node0, node1, weight)
	
	def write_edge_list(self, path_edge_list):
		edges = np.zeros((0, 3), dtype=np.float64)
		for node_id, node in self.nodes.items():
			for neighbor, weight in node.edges:
				# Avoid duplicate edges
				if node_id < neighbor.id:
					vec = np.zeros((1, 3), dtype=np.float64)
					vec[0, 0], vec[0, 1], vec[0, 2] = node_id, neighbor.id, weight
					edges = np.vstack((edges, vec))
		np.savetxt(path_edge_list, edges, fmt='%d %d %.6f')

	# Add a new node to the graph if it doesn't already exist
	def add_node(self, new_node):
		if not self.contain_node(new_node):
			self.nodes[new_node.id] = new_node

	def add_edge_undirected(self, from_node, to_node, weight):
		# Add an edge between two nodes if both nodes exist in the graph
		if self.contain_node(from_node) and self.contain_node(to_node):
			from_node.add_edge(to_node, weight)
			to_node.add_edge(from_node, weight)  # Assuming undirected graph

	# Add an edge between two nodes if both nodes exist in the graph
	def add_edge_directed(self, from_node, to_node, weight):
		if self.contain_node(from_node) and self.contain_node(to_node):
			for edge in from_node.edges:
				# Edge already exists with the same weight, do not add again
				if (edge[0].id == to_node.id) and (edge[1] == weight):
					return
				# Replace the current lighter edge with the new edge
				if (edge[0].id == to_node.id) and (edge[1] > weight):
					from_node.edges.remove(edge)  # Remove the current heavier edge
					break
				from_node.add_edge(to_node, weight)  # Add the new edge with the specified weight

	# Return the node with the given id if it exists, otherwise return None
	def get_node(self, id: int):
		if id in self.nodes:
			return self.nodes[id]
		else:
			return None

	# Return the number of nodes in the graph
	def get_num_node(self):
		return len(self.nodes)

	# Return a list of all node ids in the graph
	def get_all_id(self):
		all_id = [id for id in self.nodes.keys()]
		if all_id == []:
			return [-1]
		return all_id

	def contain_node(self, query_node):
		# Check if a node with the given id exists in the graph
		if query_node.id in self.nodes:
			return True
		else:
			return

	def check_node_connected(self, node1, node2):
		# Check if two nodes are connected using DFS
		if not self.contain_node(node1) or not self.contain_node(node2):
			return False
		visited = set()
		return self.dfs(node1, node2, visited)

	def dfs(self, current_node, target_node, visited):
		if current_node.id == target_node.id:
			return True
		visited.add(current_node.id)
		for neighbor, _ in current_node.edges:
			if neighbor.id not in visited:
				if self.dfs(neighbor, target_node, visited):
					return True
		return False

	def convert_to_gtsam_pose_graph(self):
		# Convert the base graph to a gtsam pose graph
		pose_graph = PoseGraph()
		prior_sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
		odom_sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
		for node in self.nodes.values():
			curr_pose3 = convert_vec_gtsam_pose3(node.trans, node.quat)
			# Add prior factor
			if node.id == 0:
				pose_graph.add_prior_factor(node.id, curr_pose3, prior_sigma)
			pose_graph.add_init_estimate(node.id, curr_pose3)
			# Add odometry factor
			for edge in node.edges:
				next_node = self.get_node(edge[0].id)
				next_pose3 = convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
				pose_graph.add_odometry_factor(node.id, curr_pose3, next_node.id, next_pose3, odom_sigma)
		return pose_graph			

if __name__ == "__main__":
	import sys
	import os
	path_dir = os.path.dirname(os.path.abspath(__file__))
	from base_node import BaseNode

	graph0 = BaseGraph()
	N = 10
	for id in range(N):
		graph0.add_node(BaseNode(id))
	graph0.read_edge_list(os.path.join(path_dir, '../../../../data/utils_algorithm/edge_list.txt'))
	print('Graph0:')
	print(graph0)

	graph1 = BaseGraph()
	N = 10
	for id in range(N):
		graph1.add_node(BaseNode(id))
	graph1.read_edge_list(os.path.join(path_dir, '../../../../data/utils_algorithm/edge_list.txt'))
	print('Graph1:')
	print(graph1)

	edges_A_B_weight = [
		(0, 1, 1.0),
		(1, 2, 1.0),
		(2, 3, 1.0),]
	graph0.merge(graph1, edges_A_B_weight)
	print(graph0)
	for node in graph0.nodes.values():
		print(node)
		for neighbor, weight in node.edges:
			print(f"      Neighbor ID: {neighbor.id}, Weight: {weight}")		
