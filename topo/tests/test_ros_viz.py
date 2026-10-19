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
