"""litevloc_topo.planner 的单测:plan / replan(封边)/ advance(子目标推进)。"""

import math

import pytest

from litevloc_topo.goals import KeyframeGoal
from litevloc_topo.graph import Pose2D, TopoGraph, TopoGraphError
from litevloc_topo.planner import TopoPlanner

_POSES = {
    0: Pose2D(0.0, 0.0, 0.0),
    1: Pose2D(1.0, 0.0, 0.0),
    2: Pose2D(2.0, 0.0, 0.0),
    3: Pose2D(2.0, 1.0, math.pi / 2),
}
_ADJACENCY = {
    0: ((1, 1.0),),
    1: ((0, 1.0), (2, 1.0)),
    2: ((1, 1.0), (3, 1.0)),
    3: ((2, 1.0),),
}


def _graph() -> TopoGraph:
    return TopoGraph(poses=dict(_POSES), adjacency=dict(_ADJACENCY))


def test_plan_finds_route_to_goal_keyframe() -> None:
    planner = TopoPlanner(_graph())
    route = planner.plan(Pose2D(0.0, 0.0, 0.0), KeyframeGoal(3))
    assert route.kf_path == (0, 1, 2, 3)
    assert route.length_m == pytest.approx(3.0)
    assert planner.current_route is route


def test_plan_unknown_goal_raises() -> None:
    planner = TopoPlanner(_graph())
    with pytest.raises(TopoGraphError):
        planner.plan(Pose2D(0.0, 0.0, 0.0), KeyframeGoal(99))


def test_replan_without_prior_plan_raises() -> None:
    planner = TopoPlanner(_graph())
    with pytest.raises(TopoGraphError):
        planner.replan(Pose2D(0.0, 0.0, 0.0))


def test_block_edge_forces_detour_on_replan() -> None:
    # 0-1-2 经 1 走总长 2.0,比 0-2 直连的 3.0 短;封掉 0-1 之后只能走直连那条(变长的绕路)。
    graph = TopoGraph(
        poses={0: Pose2D(0.0, 0.0, 0.0), 1: Pose2D(1.0, 0.0, 0.0), 2: Pose2D(1.0, 1.0, 0.0)},
        adjacency={
            0: ((1, 1.0), (2, 3.0)),
            1: ((0, 1.0), (2, 1.0)),
            2: ((1, 1.0), (0, 3.0)),
        },
    )
    planner = TopoPlanner(graph)
    route = planner.plan(Pose2D(0.0, 0.0, 0.0), KeyframeGoal(2))
    assert route.kf_path == (0, 1, 2)
    route2 = planner.replan(Pose2D(0.0, 0.0, 0.0), block_edge=(0, 1))
    assert route2.kf_path == (0, 2)
    assert planner.blocked == frozenset({(0, 1)})
    planner.clear_blocked()
    assert planner.blocked == frozenset()


def test_advance_steps_through_subgoals_and_finishes() -> None:
    planner = TopoPlanner(_graph())
    planner.plan(Pose2D(0.0, 0.0, 0.0), KeyframeGoal(3))
    # 机身刚好在 kf0 上,reach_m 0.1 应直接推进到 kf1
    assert planner.advance(Pose2D(0.0, 0.0, 0.0), reach_m=0.1) == 1
    assert planner.advance(Pose2D(1.0, 0.0, 0.0), reach_m=0.1) == 2
    assert planner.advance(Pose2D(2.0, 0.0, 0.0), reach_m=0.1) == 3
    assert planner.advance(Pose2D(2.0, 1.0, 0.0), reach_m=0.1) is None


def test_advance_without_plan_returns_none() -> None:
    planner = TopoPlanner(_graph())
    assert planner.advance(Pose2D(0.0, 0.0, 0.0), reach_m=1.0) is None


def test_advance_same_start_goal_finishes_immediately() -> None:
    planner = TopoPlanner(_graph())
    planner.plan(Pose2D(0.0, 0.0, 0.0), KeyframeGoal(0))
    assert planner.advance(Pose2D(0.0, 0.0, 0.0), reach_m=0.1) is None
