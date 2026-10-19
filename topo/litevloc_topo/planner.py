"""拓扑规划器:起点找最近关键帧、按目标解出关键帧、Dijkstra 规划,支持封边重规划和子目标推进。"""

import math
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

from .goals import Goal
from .graph import Pose2D, TopoGraph, TopoGraphError
from .route import Route, path_to_waypoints_with_kf
from .search import NoPathError, dijkstra


class TopoPlanner:
    """一次任务生命周期内的规划状态:封过的边一直封着,直到 ``clear_blocked``。"""

    def __init__(self, graph: TopoGraph, map_dir: Optional[Path] = None) -> None:
        self._graph = graph
        self._map_dir = map_dir
        self._blocked: FrozenSet[Tuple[int, int]] = frozenset()
        self._current_route: Optional[Route] = None
        self._goal_kf: Optional[int] = None
        self._subgoal_index = 0

    @property
    def blocked(self) -> FrozenSet[Tuple[int, int]]:
        """已经封掉的边,无向规范化成 ``(min, max)``。"""
        return self._blocked

    @property
    def current_route(self) -> Optional[Route]:
        """最近一次 ``plan`` / ``replan`` 产出的路线;没规划过是 ``None``。"""
        return self._current_route

    def block_edge(self, i: int, j: int) -> None:
        """封一条边(无向),立即生效(下次规划才会避开,不会打断当前 ``current_route``)。"""
        self._blocked = self._blocked | {(min(i, j), max(i, j))}

    def clear_blocked(self) -> None:
        """清空封掉的边。"""
        self._blocked = frozenset()

    def _plan_to(self, start_kf: int, goal_kf: int) -> Route:
        if goal_kf not in self._graph.poses:
            raise TopoGraphError(f"目标关键帧不在地图里: {goal_kf}")
        try:
            kf_path, length_m = dijkstra(self._graph, start_kf, goal_kf, blocked=self._blocked)
        except NoPathError as exc:
            raise TopoGraphError(str(exc)) from exc
        pairs = path_to_waypoints_with_kf(self._graph, kf_path)
        route = Route(
            kf_path=tuple(kf_path),
            length_m=length_m,
            waypoints=tuple(point for point, _ in pairs),
            waypoint_kfs=tuple(kf for _, kf in pairs),
        )
        self._current_route = route
        self._goal_kf = goal_kf
        self._subgoal_index = 0
        return route

    def plan(self, start: Pose2D, goal: Goal) -> Route:
        """起点 = 离 ``start`` 最近的关键帧,终点 = ``goal.resolve(...)``,按 ``self.blocked`` 规划。

        Raises:
            TopoGraphError: 目标关键帧不在地图里,或和起点不连通。
        """
        start_kf = self._graph.nearest(start.x, start.y)
        goal_kf = goal.resolve(self._graph, self._map_dir)
        return self._plan_to(start_kf, goal_kf)

    def replan(self, from_pose: Pose2D, block_edge: Optional[Tuple[int, int]] = None) -> Route:
        """先封边(如果给了),再从 ``from_pose`` 到**上一次 plan 的目标关键帧**重规划。

        Raises:
            TopoGraphError: 还没有 ``plan`` 过,或重规划不出路线。
        """
        if block_edge is not None:
            self.block_edge(*block_edge)
        if self._goal_kf is None:
            raise TopoGraphError("还没有 plan 过,不能 replan")
        start_kf = self._graph.nearest(from_pose.x, from_pose.y)
        return self._plan_to(start_kf, self._goal_kf)

    def advance(self, robot: Pose2D, reach_m: float) -> Optional[int]:
        """子目标推进:机身到当前子目标的距离 < ``reach_m`` 就前进一个,返回当前子目标关键帧。

        走完整条路线(所有关键帧都到过)返回 ``None``;还没 ``plan`` 过也返回 ``None``。
        """
        route = self._current_route
        if route is None:
            return None
        while self._subgoal_index < len(route.kf_path):
            subgoal_kf = route.kf_path[self._subgoal_index]
            pose = self._graph.poses[subgoal_kf]
            if math.hypot(robot.x - pose.x, robot.y - pose.y) < reach_m:
                self._subgoal_index += 1
                continue
            return subgoal_kf
        return None
