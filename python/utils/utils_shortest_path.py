import heapq
import unittest

def dijkstra_shortest_path(graph, start_node, goal_node):
    if not graph.contain_node(start_node) or not graph.contain_node(goal_node):
        return float('inf'), []

    distances = {node: float('inf') for _, node in graph.nodes.items()}
    distances[start_node] = 0
    previous_nodes = {node: None for _, node in graph.nodes.items()}

    pq = [(0, start_node)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in current_node.edges:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    path = []
    current_node = goal_node
    while previous_nodes[current_node] is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.append(start_node)
    path.reverse()

    return distances[goal_node], path

##### TestShortestPath #####
# Assume the Node and Graph classes are defined as follows:
class Node:
  def __init__(self, id):
    self.id = id
    self.edges = []

  def __lt__(self, other):
    return self.id < other.id

  def add_edge(self, neighbor, weight):
    self.edges.append((neighbor, weight))

class Graph:
  def __init__(self):
    self.nodes = {}

  def add_node(self, node):
    self.nodes[node.id] = node

  def contain_node(self, node):
    return node.id in self.nodes

class TestShortestPath(unittest.TestCase):
  def setUp(self):
    self.graph = Graph()

    # Create nodes
    self.node_a = Node(0)
    self.node_b = Node(1)
    self.node_c = Node(2)
    self.node_d = Node(3)
    self.node_e = Node(4)

    # Add nodes to the graph
    self.graph.add_node(self.node_a)
    self.graph.add_node(self.node_b)
    self.graph.add_node(self.node_c)
    self.graph.add_node(self.node_d)
    self.graph.add_node(self.node_e)

    # Create edges
    self.node_a.add_edge(self.node_b, 1)
    self.node_a.add_edge(self.node_c, 4)
    self.node_b.add_edge(self.node_c, 2)
    self.node_b.add_edge(self.node_d, 5)
    self.node_c.add_edge(self.node_d, 1)
    self.node_d.add_edge(self.node_e, 3)
    self.node_c.add_edge(self.node_e, 6)

  def test_shortest_path(self):
    distance, path = dijkstra_shortest_path(self.graph, self.node_a, self.node_e)
    self.assertEqual(distance, 7)
    self.assertEqual([node.id for node in path], [0, 1, 2, 3, 4])

  def test_no_path(self):
    new_node = Node(5)
    self.graph.add_node(new_node)
    distance, path = dijkstra_shortest_path(self.graph, self.node_a, new_node)
    self.assertEqual(distance, float('inf'))
    self.assertEqual(path, [self.node_a])

  def test_same_start_goal(self):
    distance, path = dijkstra_shortest_path(self.graph, self.node_a, self.node_a)
    self.assertEqual(distance, 0)
    self.assertEqual([node.id for node in path], [0])

if __name__ == '__main__':
    unittest.main()
