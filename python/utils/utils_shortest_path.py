#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from litevloc_topo.graph import Pose2D, TopoGraph
from litevloc_topo.search import NoPathError, dijkstra

def dijk_shortest_path(graph, start_node, goal_node):
    """
    Implements Dijkstra's algorithm to find the shortest path between two nodes in a graph.

    The actual search is delegated to ``litevloc_topo.search.dijkstra`` (the one shared
    Dijkstra implementation used by both OpenNavMap/nav_system and LiteVLoc); this function
    only adapts ``BaseGraph``/``BaseNode`` objects to and from that API, keeping the historical
    signature and return value (``distance``, ``[node, ...]``) unchanged for callers such as
    ``image_graph.py`` / ``point_graph.py``.
    """
    if not graph.contain_node(start_node) or not graph.contain_node(goal_node):
        return float('inf'), []

    # litevloc_topo.graph.TopoGraph only needs poses to check that a node id exists; the actual
    # edge lengths below come straight from the file weights already stored on each node's edges
    # (not recomputed from position), matching the historical behaviour of this function.
    poses = {node_id: Pose2D(0.0, 0.0, 0.0) for node_id in graph.nodes}
    adjacency = {
        node_id: tuple((neighbor.id, weight) for neighbor, weight in node.edges.values())
        for node_id, node in graph.nodes.items()
    }
    topo_graph = TopoGraph(poses=poses, adjacency=adjacency)

    try:
        node_ids, distance = dijkstra(topo_graph, start_node.id, goal_node.id)
    except NoPathError:
        return float('inf'), [start_node]

    path = [graph.get_node(node_id) for node_id in node_ids]
    return distance, path
