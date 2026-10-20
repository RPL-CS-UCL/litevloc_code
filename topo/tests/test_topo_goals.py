"""litevloc_topo.goals 的单测:KeyframeGoal / ObjectGoal / ImageGoal / LanguageGoal。"""

import json
from pathlib import Path

import numpy as np
import pytest

from litevloc_topo.goals import Goal, ImageGoal, KeyframeGoal, LanguageGoal, ObjectGoal, PoseGoal
from litevloc_topo.graph import Pose2D, TopoGraph, TopoGraphError

_GRAPH = TopoGraph(
    poses={0: Pose2D(0.0, 0.0, 0.0), 1: Pose2D(1.0, 0.0, 0.0), 2: Pose2D(2.0, 0.0, 0.0)},
    adjacency={0: ((1, 1.0),), 1: ((0, 1.0), (2, 1.0)), 2: ((1, 1.0),)},
)


def test_keyframe_goal_returns_given_id() -> None:
    assert KeyframeGoal(2).resolve(_GRAPH, None) == 2


def _write_objects(map_dir: Path) -> None:
    data = {
        "objects": [
            {"id": "obj_a", "label": "chair", "best_crop": [1, [0, 0, 10, 10]]},
            {"id": "obj_no_crop", "label": "plant"},
        ]
    }
    (map_dir / "objects.json").write_text(json.dumps(data), encoding="utf-8")


def test_object_goal_reads_best_crop(tmp_path: Path) -> None:
    _write_objects(tmp_path)
    assert ObjectGoal("obj_a").resolve(_GRAPH, tmp_path) == 1


def test_object_goal_missing_instance_raises(tmp_path: Path) -> None:
    _write_objects(tmp_path)
    with pytest.raises(TopoGraphError):
        ObjectGoal("obj_missing").resolve(_GRAPH, tmp_path)


def test_object_goal_without_best_crop_raises(tmp_path: Path) -> None:
    _write_objects(tmp_path)
    with pytest.raises(TopoGraphError):
        ObjectGoal("obj_no_crop").resolve(_GRAPH, tmp_path)


def test_object_goal_without_map_dir_raises() -> None:
    with pytest.raises(TopoGraphError):
        ObjectGoal("obj_a").resolve(_GRAPH, None)


def test_image_goal_ranked_kf_ids_picks_first_present() -> None:
    goal = ImageGoal(ranked_kf_ids=[99, 1, 0])
    assert goal.resolve(_GRAPH, None) == 1


def test_image_goal_ranked_kf_ids_all_missing_raises() -> None:
    goal = ImageGoal(ranked_kf_ids=[97, 98, 99])
    with pytest.raises(TopoGraphError):
        goal.resolve(_GRAPH, None)


def test_image_goal_descriptor_cosine_similarity() -> None:
    query = np.array([1.0, 0.0])
    map_descs = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    goal = ImageGoal(query_desc=query, map_descs=map_descs)
    assert goal.resolve(_GRAPH, None) == 1


def test_image_goal_requires_exactly_one_mode() -> None:
    with pytest.raises(TopoGraphError):
        ImageGoal()
    with pytest.raises(TopoGraphError):
        ImageGoal(ranked_kf_ids=[0], query_desc=np.array([1.0]), map_descs=np.array([[1.0]]))
    with pytest.raises(TopoGraphError):
        ImageGoal(query_desc=np.array([1.0]))


class _FixedResolver:
    def resolve_text(self, text: str) -> Goal:
        return KeyframeGoal(2)


def test_language_goal_delegates_to_resolver() -> None:
    goal = LanguageGoal("去冰箱那边", _FixedResolver())
    assert goal.resolve(_GRAPH, None) == 2


def test_pose_goal_resolves_to_nearest_keyframe() -> None:
    # (1.9, 0.0) 离 kf2(2.0, 0.0) 最近
    assert PoseGoal(1.9, 0.0, 0.0).resolve(_GRAPH, None) == 2
    assert PoseGoal(0.1, 0.0, 0.0).resolve(_GRAPH, None) == 0
