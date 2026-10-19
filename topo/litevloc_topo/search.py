"""关键帧拓扑图上的最短路与多跳邻域,全仓库唯一的 Dijkstra 实现。"""

import heapq
import math
from typing import Collection, Iterable, List, Set, Tuple

from .graph import TopoGraph


class NoPathError(Exception):
    """起点 / 终点不在图里,或两者不连通。"""


def dijkstra(
    graph: TopoGraph,
    start: int,
    goal: int,
    blocked: Collection[Tuple[int, int]] = (),
) -> Tuple[List[int], float]:
    """从 ``start`` 到 ``goal`` 的最短路(边长已经是欧氏距离或文件权重)。

    Args:
        graph: 拓扑图。
        start: 起点关键帧编号。
        goal: 终点关键帧编号。
        blocked: 要避开的边,按**无向**处理——``(i, j)`` 和 ``(j, i)`` 都封。

    Returns:
        ``(节点序列, 总长)``:节点序列含首尾;``start == goal`` 时返回 ``([start], 0.0)``。

    Raises:
        NoPathError: ``start`` / ``goal`` 不在图里,或两者不连通。
    """
    if start not in graph.poses or goal not in graph.poses:
        raise NoPathError(f"关键帧 {start} 或 {goal} 不在图里")
    if start == goal:
        return [start], 0.0
    blocked_set = {tuple(sorted(pair)) for pair in blocked}
    dist = {start: 0.0}
    prev = {}
    visited = set()
    heap: List[Tuple[float, int]] = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for neighbor, length in graph.adjacency.get(node, ()):
            if tuple(sorted((node, neighbor))) in blocked_set:
                continue
            nd = d + length
            if nd < dist.get(neighbor, math.inf):
                dist[neighbor] = nd
                prev[neighbor] = node
                heapq.heappush(heap, (nd, neighbor))
    if goal not in dist:
        raise NoPathError(f"关键帧 {start} 到 {goal} 不连通")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[goal]


def k_hop(graph: TopoGraph, seeds: Iterable[int], k: int) -> Set[int]:
    """从 ``seeds`` 出发 ≤ ``k`` 跳内的所有关键帧(含 ``seeds`` 自己)。"""
    frontier = set(seeds)
    visited = set(frontier)
    for _ in range(k):
        next_frontier: Set[int] = set()
        for node in frontier:
            for neighbor, _ in graph.adjacency.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return visited
