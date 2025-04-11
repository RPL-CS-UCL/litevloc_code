#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import heapq
import unittest

from utils.base_node import BaseNode as Node
from utils.base_graph import BaseGraph as Graph

def dijk_shortest_path(graph, start_node, goal_node):
    """
    Implements Dijkstra's algorithm to find the shortest path between two nodes in a graph. 
    """
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

        for neighbor, weight in current_node.edges.values():
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
    distance, path = dijk_shortest_path(self.graph, self.node_a, self.node_e)
    self.assertEqual(distance, 7)
    self.assertEqual([node.id for node in path], [0, 1, 2, 3, 4])

  def test_no_path(self):
    new_node = Node(5)
    self.graph.add_node(new_node)
    distance, path = dijk_shortest_path(self.graph, self.node_a, new_node)
    self.assertEqual(distance, float('inf'))
    self.assertEqual(path, [self.node_a])

  def test_same_start_goal(self):
    distance, path = dijk_shortest_path(self.graph, self.node_a, self.node_a)
    self.assertEqual(distance, 0)
    self.assertEqual([node.id for node in path], [0])

if __name__ == '__main__':
    unittest.main()
