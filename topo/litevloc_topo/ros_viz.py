"""``Route`` / ``TopoGraph`` 的 RViz 可视化;rospy 与 ROS 消息类型都在函数体内延迟 import,
模块本身不碰 ROS,不进单测主线(单测靠 ``pytest.importorskip("nav_msgs.msg")`` 跳过)。

只提供构造消息的函数,不建 publisher——发布由调用方(B7)接。
"""

import math
from typing import Optional

from .graph import TopoGraph
from .route import Route


def route_to_path_msg(route: Route, frame_id: str, stamp):
    """``Route`` 的路点序列 → ``nav_msgs/Path``。"""
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    for waypoint in route.waypoints:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = stamp
        pose_stamped.pose.position.x = waypoint.x
        pose_stamped.pose.position.y = waypoint.y
        pose_stamped.pose.orientation.z = math.sin(waypoint.yaw / 2.0)
        pose_stamped.pose.orientation.w = math.cos(waypoint.yaw / 2.0)
        msg.poses.append(pose_stamped)
    return msg


def route_to_markers(
    route: Route,
    graph: TopoGraph,
    frame_id: str,
    stamp,
    current_kf: Optional[int] = None,
):
    """路线 + 图 → ``visualization_msgs/MarkerArray``。

    内容:路线连线(绿色 LINE_STRIP)+ 途经路点(SPHERE_LIST)+ 起点(蓝色方块)+ 终点(红色带朝向
    箭头)+ 当前所在关键帧(黄色大球,``current_kf`` 给了才画)+ 每个关键帧编号(TEXT_VIEW_FACING)。
    """
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
        return ColorRGBA(r=r, g=g, b=b, a=a)

    def _header(marker: Marker) -> None:
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.action = Marker.ADD

    array = MarkerArray()
    next_id = [0]

    def _next_id() -> int:
        next_id[0] += 1
        return next_id[0] - 1

    line = Marker()
    _header(line)
    line.ns, line.id = "litevloc_topo/route", _next_id()
    line.type = Marker.LINE_STRIP
    line.scale.x = 0.05
    line.color = _color(0.0, 1.0, 0.0)
    line.points = [Point(x=w.x, y=w.y, z=0.0) for w in route.waypoints]
    array.markers.append(line)

    spheres = Marker()
    _header(spheres)
    spheres.ns, spheres.id = "litevloc_topo/waypoints", _next_id()
    spheres.type = Marker.SPHERE_LIST
    spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.12
    spheres.color = _color(0.0, 0.8, 0.2)
    spheres.points = [Point(x=w.x, y=w.y, z=0.0) for w in route.waypoints]
    array.markers.append(spheres)

    if route.waypoints:
        start = route.waypoints[0]
        start_marker = Marker()
        _header(start_marker)
        start_marker.ns, start_marker.id = "litevloc_topo/start", _next_id()
        start_marker.type = Marker.CUBE
        start_marker.pose.position.x = start.x
        start_marker.pose.position.y = start.y
        start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.2
        start_marker.color = _color(0.0, 0.0, 1.0)
        array.markers.append(start_marker)

        goal = route.waypoints[-1]
        goal_marker = Marker()
        _header(goal_marker)
        goal_marker.ns, goal_marker.id = "litevloc_topo/goal", _next_id()
        goal_marker.type = Marker.ARROW
        goal_marker.pose.position.x = goal.x
        goal_marker.pose.position.y = goal.y
        goal_marker.pose.orientation.z = math.sin(goal.yaw / 2.0)
        goal_marker.pose.orientation.w = math.cos(goal.yaw / 2.0)
        goal_marker.scale.x = 0.4
        goal_marker.scale.y = 0.08
        goal_marker.scale.z = 0.08
        goal_marker.color = _color(1.0, 0.0, 0.0)
        array.markers.append(goal_marker)

    if current_kf is not None and current_kf in graph.poses:
        pose = graph.poses[current_kf]
        current_marker = Marker()
        _header(current_marker)
        current_marker.ns, current_marker.id = "litevloc_topo/current", _next_id()
        current_marker.type = Marker.SPHERE
        current_marker.pose.position.x = pose.x
        current_marker.pose.position.y = pose.y
        current_marker.scale.x = current_marker.scale.y = current_marker.scale.z = 0.3
        current_marker.color = _color(1.0, 1.0, 0.0)
        array.markers.append(current_marker)

    for kf, pose in graph.poses.items():
        text_marker = Marker()
        _header(text_marker)
        text_marker.ns, text_marker.id = "litevloc_topo/labels", _next_id()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.pose.position.x = pose.x
        text_marker.pose.position.y = pose.y
        text_marker.pose.position.z = 0.3
        text_marker.scale.z = 0.2
        text_marker.color = _color(1.0, 1.0, 1.0)
        text_marker.text = str(kf)
        array.markers.append(text_marker)

    return array
