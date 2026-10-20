"""litevloc_topo.graph 的单测:poses.txt 读取、机身投影、邻接表、连通分量。"""

import math
from pathlib import Path

import numpy as np
import pytest

from litevloc_topo.graph import (
    Pose2D,
    TopoGraphError,
    camera_to_body,
    from_edges,
    load_topo_graph,
    read_camera_poses,
)

CAMERA_FORWARD_M = 0.1582
# 相机坐标约定(OpenCV):x 右、y 下、z 朝前;yaw=0 时相机光轴对齐任务系 +x。
_CAMERA_BASE_ROTATION = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _matrix_to_quat_wxyz(r: np.ndarray):
    """旋转矩阵转四元数(wxyz),Shepperd 方法(和 nav_localization.se3 算法一致,顺序换成 wxyz)。"""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def _poses_line(name: str, camera_xy, yaw: float) -> str:
    """造一行 poses.txt:给定相机在任务系里的 (x, y) 与 yaw,反推出"任务→相机"那行数字。"""
    rotation_task_camera = _rot_z(yaw) @ _CAMERA_BASE_ROTATION
    translation_task_camera = np.array([camera_xy[0], camera_xy[1], 0.5])
    rotation_line = rotation_task_camera.T
    translation_line = -rotation_line @ translation_task_camera
    qw, qx, qy, qz = _matrix_to_quat_wxyz(rotation_line)
    return (
        f"seq/{name}.color.jpg {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
        f"{translation_line[0]:.9f} {translation_line[1]:.9f} {translation_line[2]:.9f}"
    )


def _body_from_camera(camera_xy, yaw: float) -> Pose2D:
    bx = camera_xy[0] - CAMERA_FORWARD_M * math.cos(yaw)
    by = camera_xy[1] - CAMERA_FORWARD_M * math.sin(yaw)
    return Pose2D(bx, by, yaw)


# 0-1(很近)-2-3 一条链;4、5 各自孤立不连边。
_KEYFRAMES = {
    0: ((0.0, 0.0), 0.0),
    1: ((0.2, 0.0), 0.0),
    2: ((1.0, 0.0), 0.0),
    3: ((3.0, 0.0), math.pi / 2),
    4: ((15.0, 0.0), math.pi / 4),
    5: ((100.0, 100.0), 0.0),
}
_EDGES = ((0, 1), (1, 2), (2, 3))


def _write_test_map(tmp_path: Path) -> Path:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    lines = [_poses_line(f"{kf:06d}", xy, yaw) for kf, (xy, yaw) in sorted(_KEYFRAMES.items())]
    (map_dir / "poses.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (map_dir / "edges_trav.txt").write_text(
        "\n".join(f"{i} {j} 0.0" for i, j in _EDGES) + "\n", encoding="utf-8"
    )
    return map_dir


def test_read_camera_poses_count(tmp_path: Path) -> None:
    map_dir = _write_test_map(tmp_path)
    poses = read_camera_poses(map_dir)
    assert len(poses) == len(_KEYFRAMES)


def test_read_camera_poses_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TopoGraphError):
        read_camera_poses(tmp_path)


def test_camera_to_body_matches_forward_offset(tmp_path: Path) -> None:
    map_dir = _write_test_map(tmp_path)
    poses = read_camera_poses(map_dir)
    for kf, (xy, yaw) in _KEYFRAMES.items():
        body = camera_to_body(poses[kf], CAMERA_FORWARD_M)
        expected = _body_from_camera(xy, yaw)
        assert body.x == pytest.approx(expected.x, abs=1e-6)
        assert body.y == pytest.approx(expected.y, abs=1e-6)
        assert body.yaw == pytest.approx(expected.yaw, abs=1e-6)


def test_load_topo_graph_builds_adjacency(tmp_path: Path) -> None:
    map_dir = _write_test_map(tmp_path)
    graph = load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M)
    assert set(graph.poses) == set(_KEYFRAMES)
    assert graph.neighbors(5) == ()
    assert len(graph) == len(_KEYFRAMES)
    # 边长按机身位置欧氏距离算,不用 edges_trav.txt 里写的占位权重 0.0
    assert graph.edge_length(0, 1) > 0.0


def test_load_topo_graph_use_file_weight(tmp_path: Path) -> None:
    map_dir = _write_test_map(tmp_path)
    graph = load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M, use_file_weight=True)
    assert graph.edge_length(0, 1) == pytest.approx(0.0)


def test_load_topo_graph_missing_edges_file_raises(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    pose_line = _poses_line("000000", (0.0, 0.0), 0.0)
    (map_dir / "poses.txt").write_text(pose_line + "\n", encoding="utf-8")
    with pytest.raises(TopoGraphError):
        load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M)


def test_load_topo_graph_bad_edge_reference_raises(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    pose_line = _poses_line("000000", (0.0, 0.0), 0.0)
    (map_dir / "poses.txt").write_text(pose_line + "\n", encoding="utf-8")
    (map_dir / "edges_trav.txt").write_text("0 99 1.0\n", encoding="utf-8")
    with pytest.raises(TopoGraphError):
        load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M)


def test_load_topo_graph_use_file_weight_without_weight_column_raises(tmp_path: Path) -> None:
    map_dir = tmp_path / "map"
    map_dir.mkdir()
    lines = [_poses_line(f"{kf:06d}", xy, yaw) for kf, (xy, yaw) in sorted(_KEYFRAMES.items())]
    (map_dir / "poses.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (map_dir / "edges_trav.txt").write_text("0 1\n", encoding="utf-8")
    with pytest.raises(TopoGraphError):
        load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M, use_file_weight=True)


def test_nearest_picks_closest_body_position(tmp_path: Path) -> None:
    map_dir = _write_test_map(tmp_path)
    graph = load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M)
    # 查询点就用某个关键帧自己的机身位置,离它自己应该比离别的关键帧更近
    body0, body3 = graph.poses[0], graph.poses[3]
    assert graph.nearest(body0.x, body0.y) == 0
    assert graph.nearest(body3.x, body3.y) == 3


def test_components_groups_isolated_nodes(tmp_path: Path) -> None:
    map_dir = _write_test_map(tmp_path)
    graph = load_topo_graph(map_dir, camera_forward_m=CAMERA_FORWARD_M)
    comps = graph.components()
    assert comps[0] == [0, 1, 2, 3]
    assert comps[1] == [4]
    assert comps[2] == [5]


def test_from_edges_builds_adjacency_from_positions() -> None:
    poses = {0: Pose2D(0.0, 0.0, 0.0), 1: Pose2D(1.0, 0.0, 0.0), 2: Pose2D(1.0, 1.0, 0.0)}
    graph = from_edges(poses, [(0, 1), (1, 2)])
    assert graph.edge_length(0, 1) == pytest.approx(1.0)
    assert graph.edge_length(1, 2) == pytest.approx(1.0)
    neighbors = graph.neighbors(0)
    assert len(neighbors) == 1
    assert neighbors[0][0] == 1
    assert neighbors[0][1] == pytest.approx(1.0)


def test_edge_length_missing_edge_raises() -> None:
    poses = {0: Pose2D(0.0, 0.0, 0.0), 1: Pose2D(1.0, 0.0, 0.0)}
    graph = from_edges(poses, [])
    with pytest.raises(TopoGraphError):
        graph.edge_length(0, 1)


def test_resolve_place_dir_prefers_place_subdir_when_present(tmp_path: Path) -> None:
    from litevloc_topo.graph import resolve_place_dir

    map_dir = tmp_path / "map"
    place_dir = map_dir / "place"
    place_dir.mkdir(parents=True)
    (place_dir / "poses.txt").write_text("", encoding="utf-8")
    assert resolve_place_dir(map_dir) == place_dir


def test_resolve_place_dir_falls_back_without_place_poses(tmp_path: Path) -> None:
    from litevloc_topo.graph import resolve_place_dir

    map_dir = tmp_path / "map"
    map_dir.mkdir()
    # 没有 place/ 子目录
    assert resolve_place_dir(map_dir) == map_dir
    # 有 place/ 子目录但没有 poses.txt(比如写一半)也退回 map_dir
    (map_dir / "place").mkdir()
    assert resolve_place_dir(map_dir) == map_dir
