"""关键帧序列 → 路点序列(稀疏化 + 插点),从旧 ``nav_mapping.map_route`` 原样搬过来。"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .graph import Pose2D, TopoGraph

# 中间路点太密没有意义,和上一个保留路点相距不到这个数就丢弃(最后一个永远保留)。
MIN_WAYPOINT_SPACING_M = 0.5
# 一步最多走这么远,相距更远的两个路点之间要等分插点。
MAX_SEGMENT_M = 10.0


def sparsify(
    waypoints: Sequence[Pose2D], min_spacing_m: float = MIN_WAYPOINT_SPACING_M
) -> List[Pose2D]:
    """去掉和上一个保留路点相距不到 ``min_spacing_m`` 的中间路点;首尾永远保留。"""
    if len(waypoints) <= 2:
        return list(waypoints)
    kept = [waypoints[0]]
    for point in waypoints[1:-1]:
        last = kept[-1]
        if math.hypot(point.x - last.x, point.y - last.y) >= min_spacing_m:
            kept.append(point)
    kept.append(waypoints[-1])
    return kept


def subdivide(
    waypoints: Sequence[Pose2D], max_segment_m: float = MAX_SEGMENT_M
) -> List[Pose2D]:
    """两路点相距超过 ``max_segment_m`` 就在中间等分插点(朝向填 0,最后一个保留原朝向)。"""
    if not waypoints:
        return []
    result = [waypoints[0]]
    for point in waypoints[1:]:
        prev = result[-1]
        distance = math.hypot(point.x - prev.x, point.y - prev.y)
        segments = max(1, math.ceil(distance / max_segment_m))
        for step in range(1, segments + 1):
            ratio = step / segments
            if step == segments:
                result.append(point)
            else:
                result.append(
                    Pose2D(
                        prev.x + (point.x - prev.x) * ratio,
                        prev.y + (point.y - prev.y) * ratio,
                        0.0,
                    )
                )
    return result


def _sparsify_with_kf(
    pairs: Sequence[Tuple[Pose2D, Optional[int]]], min_spacing_m: float = MIN_WAYPOINT_SPACING_M
) -> List[Tuple[Pose2D, Optional[int]]]:
    if len(pairs) <= 2:
        return list(pairs)
    kept = [pairs[0]]
    for point, kf in pairs[1:-1]:
        last_point, _ = kept[-1]
        if math.hypot(point.x - last_point.x, point.y - last_point.y) >= min_spacing_m:
            kept.append((point, kf))
    kept.append(pairs[-1])
    return kept


def _subdivide_with_kf(
    pairs: Sequence[Tuple[Pose2D, Optional[int]]], max_segment_m: float = MAX_SEGMENT_M
) -> List[Tuple[Pose2D, Optional[int]]]:
    if not pairs:
        return []
    result = [pairs[0]]
    for point, kf in pairs[1:]:
        prev_point, _ = result[-1]
        distance = math.hypot(point.x - prev_point.x, point.y - prev_point.y)
        segments = max(1, math.ceil(distance / max_segment_m))
        for step in range(1, segments + 1):
            ratio = step / segments
            if step == segments:
                result.append((point, kf))
            else:
                result.append(
                    (
                        Pose2D(
                            prev_point.x + (point.x - prev_point.x) * ratio,
                            prev_point.y + (point.y - prev_point.y) * ratio,
                            0.0,
                        ),
                        None,
                    )
                )
    return result


def path_to_waypoints_with_kf(
    graph: TopoGraph, path: Sequence[int]
) -> List[Tuple[Pose2D, Optional[int]]]:
    """同 ``path_to_waypoints``,但每个路点附带它对应的关键帧编号(插出来的中间点填 ``None``)。

    给 ``planner.TopoPlanner`` 组装 ``Route`` 用:``RouteWalker`` 要能按关键帧编号封边重规划,
    所以要能从路点反查关键帧。
    """
    pairs = [(Pose2D(graph.poses[kf].x, graph.poses[kf].y, 0.0), kf) for kf in path]
    pairs[-1] = (graph.poses[path[-1]], path[-1])
    return _subdivide_with_kf(_sparsify_with_kf(pairs))


def path_to_waypoints(graph: TopoGraph, path: Sequence[int]) -> List[Pose2D]:
    """关键帧序列 → 路点:中间路点朝向填 0,最后一个用目标关键帧真实朝向,再稀疏化 + 插点。"""
    return [point for point, _ in path_to_waypoints_with_kf(graph, path)]


@dataclass(frozen=True)
class Route:
    """一次规划的结果。

    Attributes:
        kf_path: 规划出的关键帧序列(含首尾)。
        length_m: 这条关键帧路径的总长度(米)。
        waypoints: 稀疏化 + 插点后的路点序列。
        waypoint_kfs: 每个路点对应的关键帧编号;插出来的中间点是 ``None``。
    """

    kf_path: Tuple[int, ...]
    length_m: float
    waypoints: Tuple[Pose2D, ...]
    waypoint_kfs: Tuple[Optional[int], ...]
