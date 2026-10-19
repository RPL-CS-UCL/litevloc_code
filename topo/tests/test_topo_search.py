"""litevloc_topo.search 的单测。

前三个用例从 LiteVLoc 旧 ``utils/utils_shortest_path.py`` 里的 ``TestShortestPath`` 搬过来,
数值不变(a→e 距离 7、路径 [0,1,2,3,4];不连通;起终同点),改成测 ``litevloc_topo`` 的新 API。
"""

import pytest

from litevloc_topo.graph import Pose2D, TopoGraph
from litevloc_topo.search import NoPathError, dijkstra, k_hop

_ZERO = Pose2D(0.0, 0.0, 0.0)


def _make_graph() -> TopoGraph:
    """a=0 b=1 c=2 d=3 e=4,边权重照抄旧 TestShortestPath.setUp。"""
    edges = {
        (0, 1): 1.0,
        (0, 2): 4.0,
        (1, 2): 2.0,
        (1, 3): 5.0,
        (2, 3): 1.0,
        (3, 4): 3.0,
        (2, 4): 6.0,
    }
    poses = {i: _ZERO for i in range(5)}
    neighbors = {i: [] for i in range(5)}
    for (i, j), w in edges.items():
        neighbors[i].append((j, w))
        neighbors[j].append((i, w))
    adjacency = {i: tuple(v) for i, v in neighbors.items()}
    return TopoGraph(poses=poses, adjacency=adjacency)


def test_dijkstra_shortest_path_distance_and_order() -> None:
    graph = _make_graph()
    path, distance = dijkstra(graph, 0, 4)
    assert distance == pytest.approx(7.0)
    assert path == [0, 1, 2, 3, 4]


def test_dijkstra_no_path_raises() -> None:
    graph = _make_graph()
    poses = dict(graph.poses)
    poses[5] = _ZERO
    adjacency = dict(graph.adjacency)
    adjacency[5] = ()
    graph = TopoGraph(poses=poses, adjacency=adjacency)
    with pytest.raises(NoPathError):
        dijkstra(graph, 0, 5)


def test_dijkstra_same_start_goal() -> None:
    graph = _make_graph()
    path, distance = dijkstra(graph, 0, 0)
    assert distance == pytest.approx(0.0)
    assert path == [0]


def test_dijkstra_unknown_node_raises() -> None:
    graph = _make_graph()
    with pytest.raises(NoPathError):
        dijkstra(graph, 0, 999)


def test_dijkstra_blocked_edge_forces_detour() -> None:
    poses = {0: _ZERO, 1: _ZERO, 2: _ZERO}
    adjacency = {0: ((1, 1.0), (2, 5.0)), 1: ((0, 1.0), (2, 1.0)), 2: ((1, 1.0), (0, 5.0))}
    graph = TopoGraph(poses=poses, adjacency=adjacency)
    path, distance = dijkstra(graph, 0, 2)
    assert path == [0, 1, 2]
    assert distance == pytest.approx(2.0)
    # blocked 按无向处理,(1, 0) 和 (0, 1) 效果一样
    path, distance = dijkstra(graph, 0, 2, blocked=[(1, 0)])
    assert path == [0, 2]
    assert distance == pytest.approx(5.0)


def test_k_hop_returns_nodes_within_distance() -> None:
    poses = {i: _ZERO for i in range(5)}
    adjacency = {
        0: ((1, 1.0),),
        1: ((0, 1.0), (2, 1.0)),
        2: ((1, 1.0), (3, 1.0)),
        3: ((2, 1.0), (4, 1.0)),
        4: ((3, 1.0),),
    }
    graph = TopoGraph(poses=poses, adjacency=adjacency)
    assert k_hop(graph, [0], 0) == {0}
    assert k_hop(graph, [0], 1) == {0, 1}
    assert k_hop(graph, [0], 2) == {0, 1, 2}
    assert k_hop(graph, [2], 10) == {0, 1, 2, 3, 4}
