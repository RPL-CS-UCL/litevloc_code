import os
import numpy as np
from pathlib import Path

class BaseGraph:
	# Initialize an empty dictionary to store nodes
	def __init__(self, map_root: Path, edge_type: str):
		self.map_root = map_root
		map_root.mkdir(exist_ok=True, parents=True)
		
		self.edge_type = edge_type
		
		# Use dict() to ensure the keys (node.id) are uniques
		self._nodes = {}

	def __str__(self):
		num_edge = 0
		for node_id, node in self.nodes.items(): 
			num_edge += len(node.edges)
		out_str = f"Graph has {len(self.nodes)} nodes with {num_edge} edges"
		return out_str

	@property
	def nodes(self):
		return self._nodes

	def set_node(self, new_nodes: dict):
		self._nodes = new_nodes

	def remove_node_list(self, node_list: list):
		for node in node_list:
			self.remove_node(node)

	def remove_node(self, node):
		if self.contain_node(node):
			self.nodes.pop(node.id)

	def remove_invalid_edges(self, nodes_to_remove: list):
		for node in self.nodes.values():
			for node_rm in nodes_to_remove:
				if node_rm.id in node.edges:
					node.edges.pop(node_rm.id)
					# print(f"Removed edges {node.id} -> {node_rm.id} from graph")

	def read_edge_list(self, edge_list_path: Path):
		if edge_list_path.exists():
			# list of edges [node_a.id, node_b.id, weight]
			edges_A_B_weight = np.loadtxt(str(edge_list_path), dtype=float)
			for edge in edges_A_B_weight:
				node_id0, node_id1 = int(edge[0]), int(edge[1])
				weight = edge[2]
				if (self.get_node(node_id0) is not None) and (self.get_node(node_id1) is not None):
					node0 = self.get_node(node_id0)
					node1 = self.get_node(node_id1)
					self.add_edge_undirected(node0, node1, weight)
		else:
			print(f"Edge list {str(edge_list_path)} file not found")
		
	def write_edge_list(self, edge_list_path: Path):
		edges = np.zeros((0, 3), dtype=np.float64)
		# Remove invalid edges with nodes which are previously removed
		for node in self.nodes.values():
			for neighbor, weight in node.edges.values():
				# Avoid duplicate edges
				if node.id < neighbor.id:
					vec = np.zeros((1, 3), dtype=np.float64)
					vec[0, 0], vec[0, 1], vec[0, 2] = node.id, neighbor.id, weight
					edges = np.vstack((edges, vec))				
			
		np.savetxt(str(edge_list_path), edges, fmt='%d %d %.6f')

	# Add a new node to the graph if it doesn't already exist
	def add_node(self, new_node):
		if not self.contain_node(new_node):
			self.nodes[new_node.id] = new_node
			# print(f'Adding {new_node.id} into graph')

	def add_inter_edges(
		self, 
		edges: list, # list of [node_a, node_b, T_rel, weight]
		weight_func
	):
		for edge in edges:
			weight = weight_func(edge)
			self.add_edge_undirected(edge[0], edge[1], weight)
	
	def add_edge_undirected(self, from_node, to_node, weight):
		# Add an edge between two nodes if both nodes exist in the graph
		# In some cases that from_node or to_node are removed 
		if self.contain_node(from_node) and self.contain_node(to_node):
			from_node.add_edge(to_node, weight)
			to_node.add_edge(from_node, weight)  # Assuming undirected graph

	# Add an edge between two nodes if both nodes exist in the graph
	def add_edge_directed(self, from_node, to_node, weight):
		if self.contain_node(from_node) and self.contain_node(to_node):
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

	def get_max_node_id(self):
		return max(self.get_all_id())

	def contain_node(self, query_node):
		# Check if a node with the given id exists in the graph
		if query_node.id in self.nodes:
			return True
		else:
			return False

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
		for neighbor, _ in current_node.edges.values():
			if neighbor.id not in visited:
				if self.dfs(neighbor, target_node, visited):
					return True
		return False

	def print_edges(self):
		for node in self.nodes.values():
			print(f"Node ID: {node.id}")
			for neighbor, weight in node.edges.values():
				print(f"      Neighbor ID: {neighbor.id}, Weight: {weight}")

	def find_connected_components(self):
		"""
		Return:
			sorted_components: [component1, component2, ...]
				len(component1) >= len(component2) >= ...
				component1: [node1, node2, ...] with sorted node id (ascending)
					node1.id < node2.id < ...
		"""
		visited = set()
		sorted_components = []
		for node in self.nodes.values():
			if node.id not in visited:
				stack = [node]
				visited.add(node.id)
				component = []
				while stack:
					current = stack.pop()
					component.append(current)
					for neighbor, _ in current.edges.values():
						if neighbor.id not in visited:
							visited.add(neighbor.id)
							stack.append(neighbor)
				
				component.sort(key=lambda x: x.id)
				sorted_components.append(component)
		
		sorted_components.sort(key=lambda x: -len(x))
		return sorted_components

	def get_disconnected_subgraphs(self):
		sorted_components = self.find_connected_components()
		subgraphs = []
		for id, component in enumerate(sorted_components):
			# Create subgraph, but use reference operation to store nodes
			subgraph = type(self)(self.map_root/f"submap_disc_{id}", self.edge_type)
			for node in component:
				subgraph.add_node(node)
			subgraphs.append(subgraph)

		return subgraphs


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
