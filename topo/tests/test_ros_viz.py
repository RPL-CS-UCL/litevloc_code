"""litevloc_topo.ros_viz 的单测。

只在系统 Python 装了 nav_msgs 时跑,没有 ROS 的纯 numpy 环境自动跳过。
"""

import pytest

pytest.importorskip("nav_msgs.msg")

from litevloc_topo.graph import Pose2D, TopoGraph  # noqa: E402
from litevloc_topo.ros_viz import route_to_markers, route_to_path_msg  # noqa: E402
from litevloc_topo.route import Route  # noqa: E402


def _route() -> Route:
    waypoints = (Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0), Pose2D(2.0, 0.0, 1.0))
    return Route(kf_path=(0, 1), length_m=2.0, waypoints=waypoints, waypoint_kfs=(0, None, 1))


def test_route_to_path_msg_has_one_pose_per_waypoint() -> None:
    msg = route_to_path_msg(_route(), frame_id="map", stamp=0)
    assert len(msg.poses) == 3
    assert msg.header.frame_id == "map"


def test_route_to_markers_non_empty() -> None:
    graph = TopoGraph(poses={0: Pose2D(0.0, 0.0, 0.0), 1: Pose2D(2.0, 0.0, 1.0)}, adjacency={})
    markers = route_to_markers(_route(), graph, frame_id="map", stamp=0, current_kf=0)
    assert len(markers.markers) > 0
    # 两个关键帧应该各有一个编号 TEXT 标记
    text_markers = [m for m in markers.markers if m.ns == "litevloc_topo/labels"]
    assert len(text_markers) == len(graph.poses)
    # next_index 没给 → 旧行为:整条线一个 namespace,一个绿色
    route_markers = [m for m in markers.markers if m.ns == "litevloc_topo/route"]
    assert len(route_markers) == 1
    assert len(route_markers[0].points) == 3


def test_route_to_markers_next_index_splits_walked_and_remaining() -> None:
    """``next_index`` 给了(M5-B7):路线拆成"走过的"灰线 + "没走的"绿线,首尾相接不断。"""
    graph = TopoGraph(poses={0: Pose2D(0.0, 0.0, 0.0), 1: Pose2D(2.0, 0.0, 1.0)}, adjacency={})
    markers = route_to_markers(_route(), graph, frame_id="map", stamp=0, next_index=1)

    walked = [m for m in markers.markers if m.ns == "litevloc_topo/route_walked"]
    remaining = [m for m in markers.markers if m.ns == "litevloc_topo/route_remaining"]
    assert len(walked) == 1 and len(remaining) == 1
    assert len(walked[0].points) == 2  # waypoints[0:2]:走过的第一段 + 分界点
    assert len(remaining[0].points) == 2  # waypoints[1:3]:分界点 + 没走的最后一段
    # 两条线在分界点相接,不该有缝
    assert (walked[0].points[-1].x, walked[0].points[-1].y) == (
        remaining[0].points[0].x,
        remaining[0].points[0].y,
    )
    # 旧的单色 namespace 不应该再出现
    assert not [m for m in markers.markers if m.ns == "litevloc_topo/route"]


def test_route_to_markers_next_index_zero_has_no_walked_line() -> None:
    """一段都没走完(``next_index=0``)时,"走过的"那条线不到 2 个点,不画。"""
    graph = TopoGraph(poses={0: Pose2D(0.0, 0.0, 0.0)}, adjacency={})
    markers = route_to_markers(_route(), graph, frame_id="map", stamp=0, next_index=0)
    assert not [m for m in markers.markers if m.ns == "litevloc_topo/route_walked"]
    remaining = [m for m in markers.markers if m.ns == "litevloc_topo/route_remaining"]
    assert len(remaining) == 1 and len(remaining[0].points) == 3
