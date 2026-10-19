"""litevloc_topo.route 的单测。

稀疏化 / 插点 / 关键帧路径转路点,逻辑照旧 map_route 的 _sparsify / _subdivide。
"""

import math

import pytest

from litevloc_topo.graph import Pose2D, TopoGraph
from litevloc_topo.route import (
    MAX_SEGMENT_M,
    MIN_WAYPOINT_SPACING_M,
    path_to_waypoints,
    path_to_waypoints_with_kf,
    sparsify,
    subdivide,
)


def test_sparsify_drops_close_middle_points() -> None:
    points = [Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.0, 0.0), Pose2D(5.0, 0.0, 1.0)]
    assert sparsify(points) == [points[0], points[-1]]


def test_sparsify_keeps_far_middle_points() -> None:
    points = [Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0), Pose2D(5.0, 0.0, 1.0)]
    assert sparsify(points) == points


def test_sparsify_custom_spacing() -> None:
    points = [Pose2D(0.0, 0.0, 0.0), Pose2D(0.6, 0.0, 0.0), Pose2D(5.0, 0.0, 1.0)]
    assert sparsify(points, min_spacing_m=MIN_WAYPOINT_SPACING_M) == points
    assert sparsify(points, min_spacing_m=1.0) == [points[0], points[-1]]


def test_sparsify_two_or_fewer_points_untouched() -> None:
    points = [Pose2D(0.0, 0.0, 0.0), Pose2D(0.01, 0.0, 0.0)]
    assert sparsify(points) == points


def test_subdivide_inserts_midpoints_over_max_segment() -> None:
    points = [Pose2D(0.0, 0.0, 0.0), Pose2D(25.0, 0.0, 1.0)]
    result = subdivide(points)
    assert len(result) == 4  # 25 / 10 -> 3 段,4 个点
    for a, b in zip(result[:-1], result[1:]):
        assert math.hypot(b.x - a.x, b.y - a.y) <= MAX_SEGMENT_M + 1e-9
    assert result[-1].yaw == pytest.approx(1.0)
    for point in result[1:-1]:
        assert point.yaw == 0.0


def test_subdivide_short_segment_untouched() -> None:
    points = [Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 1.0)]
    assert subdivide(points) == points


def test_subdivide_empty_input() -> None:
    assert subdivide([]) == []


def test_path_to_waypoints_keeps_goal_orientation_and_sparsifies() -> None:
    poses = {
        0: Pose2D(0.0, 0.0, 0.0),
        1: Pose2D(0.2, 0.0, 0.0),
        2: Pose2D(1.0, 0.0, 0.0),
        3: Pose2D(3.0, 0.0, math.pi / 2),
    }
    adjacency = {
        0: ((1, 0.2),),
        1: ((0, 0.2), (2, 0.8)),
        2: ((1, 0.8), (3, 2.0)),
        3: ((2, 2.0),),
    }
    graph = TopoGraph(poses=poses, adjacency=adjacency)
    waypoints = path_to_waypoints(graph, [0, 1, 2, 3])
    # kf1 离 kf0 只有 0.2 米,小于 MIN_WAYPOINT_SPACING_M,应该被稀疏掉
    assert all(math.hypot(p.x - 0.2, p.y - 0.0) > 1e-6 for p in waypoints[:-1])
    assert waypoints[-1].x == pytest.approx(3.0)
    assert waypoints[-1].yaw == pytest.approx(math.pi / 2)
    for point in waypoints[:-1]:
        assert point.yaw == 0.0


def test_path_to_waypoints_with_kf_tracks_source_keyframe() -> None:
    poses = {
        0: Pose2D(0.0, 0.0, 0.0),
        1: Pose2D(1.0, 0.0, 0.0),
        2: Pose2D(25.0, 0.0, math.pi / 2),
    }
    adjacency = {0: ((1, 1.0),), 1: ((0, 1.0), (2, 24.0)), 2: ((1, 24.0),)}
    graph = TopoGraph(poses=poses, adjacency=adjacency)
    pairs = path_to_waypoints_with_kf(graph, [0, 1, 2])
    # 0 -> 1 距离够,不会被稀疏掉;1 -> 2 距离 24 米超过 MAX_SEGMENT_M(10 米),插两个中间点(kf=None)
    kfs = [kf for _, kf in pairs]
    assert kfs == [0, 1, None, None, 2]
