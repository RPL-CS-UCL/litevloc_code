import os
import numpy as np

import heapq

from utils.utils_image import load_image

class ImageGraphLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(image_graph_path, image_size, normalized=False, num_sample=1, num_load=10000):
		image_graph = ImageGraph()

		poses_w_cam = np.loadtxt(os.path.join(image_graph_path, 'camera_pose_gt.txt'))
		for i in range(0, 
								 	 min(poses_w_cam.shape[0], num_load * num_sample), 
									 num_sample):
			img_path = os.path.join(image_graph_path, 'rgb', f'{i:06}.png')
			image = load_image(img_path, image_size, normalized=normalized)

			pose_w_cam = poses_w_cam[i, :]
			time = pose_w_cam[0]
			t_w_cam = pose_w_cam[1:4]
			quat_w_cam = np.roll(pose_w_cam[4:], 1) # [qw, qx, qy, qz]

			node = Node(i, image, f'image node {i}', time, t_w_cam, quat_w_cam, img_path)
			image_graph.add_node(node)

			if i / num_sample > num_load:
				break

		return image_graph
	
class Node:
	def __init__(self, id, image, descriptor, time, t_w_cam, quat_w_cam, img_path):
		self.id = id
		self.image = image
		self.descriptor = descriptor

		self.time = time
		self.t_w_cam = t_w_cam
		self.quat_w_cam = quat_w_cam

		self.img_path = img_path

		self.edges = []

	def add_edge(self, neighbor, weight):
		self.edges.append((neighbor, weight))

	def set_descriptor(self, descriptor):
		self.descriptor = descriptor

	def get_descriptor(self):
		return self.descriptor

class ImageGraph:
	def __init__(self):
		self.nodes = {}

	def add_node(self, new_node):
		if new_node.id not in self.nodes:
			self.nodes[new_node.id] = new_node

	def add_edge(self, from_node, to_node, weight):
		if from_node in self.nodes and to_node in self.nodes:
			self.nodes[from_node].add_edge(to_node, weight)
			self.nodes[to_node].add_edge(from_node, weight)  # Assuming undirected graph

	def get_node(self, id):
		if id in self.nodes:
			return self.nodes[id]
		else:
			return None
		
	def get_num_node(self):
		return len(self.nodes)

	def get_all_id(self):
		all_id = [id for id in self.nodes.keys()]
		return all_id

	def shortest_path(self, start, end):
		if start not in self.nodes or end not in self.nodes:
			return float('inf'), []

		distances = {node: float('inf') for node in self.nodes}
		distances[start] = 0
		priority_queue = [(0, start)]
		previous_nodes = {node: None for node in self.nodes}

		while priority_queue:
			current_distance, current_node = heapq.heappop(priority_queue)

			if current_distance > distances[current_node]:
				continue

			for neighbor, weight in self.nodes[current_node].edges:
				distance = current_distance + weight

				if distance < distances[neighbor]:
					distances[neighbor] = distance
					previous_nodes[neighbor] = current_node
					heapq.heappush(priority_queue, (distance, neighbor))

		path = []
		current_node = end
		while previous_nodes[current_node] is not None:
			path.append(current_node)
			current_node = previous_nodes[current_node]
		path.append(start)
		path.reverse()

		return distances[end], path

	def find_connection(self, node):
		if node in self.nodes:
			return [neighbor for neighbor, _ in self.nodes[node].edges]
		else:
			return []

	def build_connection(self, node, nearby_node, weight):
		if node in self.nodes and nearby_node in self.nodes:
			self.add_edge(node, nearby_node, weight)

class TestImageGraph():
	def __init__(self):
		pass
	
	def run_test(self):
		# Initialize the image graph
		graph = ImageGraph()

		# Add nodes to the graph
		graph.add_node(Node(1, None, "descriptor_1", 0, np.zeros((1, 3)), np.zeros((1, 4)), 'tmp.png'))
		graph.add_node(Node(2, None, "descriptor_2", 0, np.zeros((1, 3)), np.zeros((1, 4)), 'tmp.png'))
		graph.add_node(Node(3, None, "descriptor_3", 0, np.zeros((1, 3)), np.zeros((1, 4)), 'tmp.png'))
		graph.add_node(Node(4, None, "descriptor_4", 0, np.zeros((1, 3)), np.zeros((1, 4)), 'tmp.png'))

		# Add edges between the nodes with weights
		graph.add_edge(1, 2, 1.0)
		graph.add_edge(2, 3, 2.0)
		graph.add_edge(3, 4, 1.0)
		graph.add_edge(1, 4, 4.0)

		# Get the image descriptor of a specific node
		node = graph.get_node(2)
		print(f"Image Descriptor of Node 2: {node.descriptor}")

		# Find the shortest path from node 1 to node 4
		distance, path = graph.shortest_path(1, 4)
		print(f"Shortest Path from Node 1 to Node 4: {path} with distance {distance}")

		# Find all connections of a specific node
		connections = graph.find_connection(2)
		print(f"Connections of Node 2: {connections}")

		# Build a new connection between two nodes
		graph.build_connection(2, 4, 2.0)

		# Verify the new connection
		connections = graph.find_connection(2)
		print(f"Connections of Node 2 after adding a new connection: {connections}")

		# Find the shortest path again from node 1 to node 4 after adding new connection
		distance, path = graph.shortest_path(1, 4)
		print(f"New Shortest Path from Node 1 to Node 4: {path} with distance {distance}")

if __name__ == '__main__':
	test_image_graph = TestImageGraph()
	test_image_graph.run_test()