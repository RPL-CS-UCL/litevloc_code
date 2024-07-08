import heapq
from collections import defaultdict

class Node:
  def __init__(self, id, image_descriptor):
      self.id = id
      self.image_descriptor = image_descriptor
      self.edges = []

  def add_edge(self, neighbor, weight):
      self.edges.append((neighbor, weight))

class TopologicalGraph:
  def __init__(self):
      self.nodes = {}

  def add_node(self, id, image_descriptor):
      if id not in self.nodes:
          self.nodes[id] = Node(id, image_descriptor)

  def add_edge(self, from_node, to_node, weight):
      if from_node in self.nodes and to_node in self.nodes:
          self.nodes[from_node].add_edge(to_node, weight)
          self.nodes[to_node].add_edge(from_node, weight)  # Assuming undirected graph

  def get_image(self, id):
      if id in self.nodes:
          return self.nodes[id].image_descriptor
      else:
          return None

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

def main():
  # Initialize the topological graph
  graph = TopologicalGraph()

  # Add nodes to the graph
  graph.add_node(1, "descriptor_1")
  graph.add_node(2, "descriptor_2")
  graph.add_node(3, "descriptor_3")
  graph.add_node(4, "descriptor_4")

  # Add edges between the nodes with weights
  graph.add_edge(1, 2, 1.0)
  graph.add_edge(2, 3, 2.0)
  graph.add_edge(3, 4, 1.0)
  graph.add_edge(1, 4, 4.0)

  # Get the image descriptor of a specific node
  image_descriptor = graph.get_image(2)
  print(f"Image Descriptor of Node 2: {image_descriptor}")

  # Find the shortest path from node 1 to node 4
  distance, path = graph.shortest_path(1, 4)
  print(f"Shortest Path from Node 1 to Node 4: {path} with distance {distance}")

  # Find all connections of a specific node
  connections = graph.find_connection(2)
  print(f"Connections of Node 2: {connections}")

  # Build a new connection between two nodes
  graph.build_connection(2, 4, 3.0)

  # Verify the new connection
  connections = graph.find_connection(2)
  print(f"Connections of Node 2 after adding a new connection: {connections}")

  # Find the shortest path again from node 1 to node 4 after adding new connection
  distance, path = graph.shortest_path(1, 4)
  print(f"New Shortest Path from Node 1 to Node 4: {path} with distance {distance}")

if __name__ == '__main__':
    main()