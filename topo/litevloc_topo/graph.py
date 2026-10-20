"""关键帧拓扑图:机身平面位姿 + 邻接表。

地图目录里只读两个文件:

- ``poses.txt``:每行 ``<图片名> qw qx qy qz tx ty tz`` = 任务系 → 相机的变换(四元数 wxyz
  顺序 + 平移);取逆得到相机在任务系(或叫地图系)里的位姿,行号(从 0 起)就是关键帧编号。
- ``edges_<edge_type>.txt``:每行 ``i j w``,i / j 是关键帧编号,边无向;边长默认按两关键帧
  **机身**平面位置的欧氏距离算,``w`` 只在 ``use_file_weight=True`` 时才用。

本模块只做平面(x、y、朝向):相机光轴是相机系 +z,投影到地图系 xy 平面取 ``atan2`` 就是关键帧
朝向;相机装在机身中心前方 ``camera_forward_m`` 米,机身位置 = 相机位置 − camera_forward_m ×
(cos yaw, sin yaw)。``camera_forward_m`` 是机器人参数,必须由调用方传入,这里不写死。
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Tuple

import numpy as np


class Pose2D(NamedTuple):
    """平面位姿(米、弧度)。字段名和 ``nav_skills.point_nav.Pose2D`` 一致,但互不依赖。"""

    x: float
    y: float
    yaw: float


class TopoGraphError(Exception):
    """地图目录读不出来、边文件格式错,或边引用了不存在的关键帧。"""


def _quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数(wxyz)转 3x3 旋转矩阵;输入不要求单位化,这里先归一化一次。"""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise TopoGraphError("四元数范数接近 0,不是合法旋转")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def resolve_place_dir(map_dir: Path) -> Path:
    """地点图(place graph)目录:``map_dir/place`` 存在且有 ``poses.txt`` 就用它,否则退回
    ``map_dir`` 本身(M5-B9:旧图没有分出地点图时,直接拿场景图当地点图用,不用重建)。

    Args:
        map_dir: 场景图目录(建图产物的那一层,不是地点图)。

    Returns:
        该用来做 LiteVLoc 定位 / 规划的那个目录。
    """
    place_dir = Path(map_dir) / "place"
    if (place_dir / "poses.txt").is_file():
        return place_dir
    return Path(map_dir)


def read_camera_poses(map_dir: Path) -> List[np.ndarray]:
    """读 ``poses.txt``,逐行算出相机在地图系里的 4x4 位姿(已取逆);行号即关键帧编号。

    空行 / ``#`` 开头跳过。数值与 ``nav_mapping.quality_files.read_keyframes`` 一致(同样的
    "读四元数 wxyz + 平移、拼矩阵、取逆"),公式在本函数里重写了一遍,不依赖 nav_system / scipy。

    Args:
        map_dir: 地图产物目录(含 ``poses.txt`` 的那一层)。

    Returns:
        每个关键帧在地图系里的 4x4 位姿(``numpy.ndarray``),按行号顺序排列。

    Raises:
        TopoGraphError: 文件缺失,或某一行格式 / 数值不对。
    """
    path = map_dir / "poses.txt"
    if not path.is_file():
        raise TopoGraphError(f"找不到 poses.txt: {path}")
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise TopoGraphError(f"poses.txt 行格式错误: {line!r}")
        try:
            qw, qx, qy, qz, x, y, z = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise TopoGraphError(f"poses.txt 行数值错误: {line!r}") from exc
        matrix = np.eye(4)
        matrix[:3, :3] = _quat_wxyz_to_matrix(qw, qx, qy, qz)
        matrix[:3, 3] = [x, y, z]
        poses.append(np.linalg.inv(matrix))
    return poses


def camera_to_body(camera_in_map: np.ndarray, camera_forward_m: float) -> Pose2D:
    """相机在地图系里的 4x4 位姿 → 机身平面位姿(照搬旧 ``map_route._camera_pose_to_body``)。

    Args:
        camera_in_map: 相机在地图系里的 4x4 位姿(``read_camera_poses`` 的一项)。
        camera_forward_m: 相机装在机身中心前方多少米。
    """
    cam_x, cam_y = camera_in_map[0, 3], camera_in_map[1, 3]
    forward = camera_in_map[:3, 2]  # 相机光轴 = 相机系 +z,这里已经是地图系下的方向向量
    yaw = math.atan2(forward[1], forward[0])
    body_x = cam_x - camera_forward_m * math.cos(yaw)
    body_y = cam_y - camera_forward_m * math.sin(yaw)
    return Pose2D(float(body_x), float(body_y), float(yaw))


def _adjacency_from_weighted_edges(
    poses: Dict[int, Pose2D], edges: Iterable[Tuple[int, int, float]]
) -> Dict[int, Tuple[Tuple[int, float], ...]]:
    neighbors: Dict[int, List[Tuple[int, float]]] = {kf: [] for kf in poses}
    for i, j, weight in edges:
        if i not in poses or j not in poses:
            raise TopoGraphError(f"边引用了不存在的关键帧: {i} / {j}")
        neighbors[i].append((j, weight))
        neighbors[j].append((i, weight))
    return {kf: tuple(v) for kf, v in neighbors.items()}


@dataclass(frozen=True)
class TopoGraph:
    """关键帧机身平面位姿 + 邻接表(边长默认是两端机身位置的欧氏距离)。"""

    poses: Dict[int, Pose2D] = field(default_factory=dict)
    adjacency: Dict[int, Tuple[Tuple[int, float], ...]] = field(default_factory=dict)

    def neighbors(self, kf: int) -> Tuple[Tuple[int, float], ...]:
        """``kf`` 的邻居列表,每项是 ``(邻居关键帧, 边长)``。"""
        return self.adjacency.get(kf, ())

    def edge_length(self, i: int, j: int) -> float:
        """``i``、``j`` 之间的边长;两者之间没有边抛 ``TopoGraphError``。"""
        for neighbor, length in self.adjacency.get(i, ()):
            if neighbor == j:
                return length
        raise TopoGraphError(f"关键帧 {i} 和 {j} 之间没有边")

    def nearest(self, x: float, y: float) -> int:
        """机身平面位置里离 ``(x, y)`` 最近的关键帧;平手取 ``min`` 的默认行为(先出现的那个)。"""
        if not self.poses:
            raise TopoGraphError("图里没有关键帧")
        return min(
            self.poses, key=lambda kf: math.hypot(self.poses[kf].x - x, self.poses[kf].y - y)
        )

    def components(self) -> List[List[int]]:
        """连通分量列表;每个分量内部按关键帧编号升序,分量之间按最小编号排序。"""
        visited = set()
        comps: List[List[int]] = []
        for kf in self.poses:
            if kf in visited:
                continue
            stack = [kf]
            visited.add(kf)
            comp: List[int] = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor, _ in self.adjacency.get(node, ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            comp.sort()
            comps.append(comp)
        comps.sort(key=lambda comp: comp[0])
        return comps

    def __len__(self) -> int:
        return len(self.poses)


def load_topo_graph(
    map_dir: Path,
    camera_forward_m: float,
    edge_type: str = "trav",
    use_file_weight: bool = False,
) -> TopoGraph:
    """读 ``poses.txt`` + ``edges_<edge_type>.txt``,算出每个关键帧的机身位姿与邻接表。

    Args:
        map_dir: 地图产物目录。
        camera_forward_m: 相机装在机身中心前方多少米,机器人参数,由调用方传入(nav_system 传
            ``CAMERA_FORWARD_M = 0.1582``)。
        edge_type: 边文件后缀,默认 ``trav``(对应 ``edges_trav.txt``)。
        use_file_weight: True 时用边文件第三列当边长(没有第三列报错);默认 False,边长按两端
            机身位置的欧氏距离算,不用文件里的 ``w``。

    Returns:
        关键帧拓扑图。

    Raises:
        TopoGraphError: 目录缺文件,边文件格式错,或边引用了不存在的关键帧。
    """
    camera_poses = read_camera_poses(map_dir)
    poses: Dict[int, Pose2D] = {
        index: camera_to_body(pose, camera_forward_m) for index, pose in enumerate(camera_poses)
    }
    edges_path = map_dir / f"edges_{edge_type}.txt"
    if not edges_path.is_file():
        raise TopoGraphError(f"找不到 edges_{edge_type}.txt: {edges_path}")
    edges: List[Tuple[int, int, float]] = []
    for line in edges_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise TopoGraphError(f"edges_{edge_type}.txt 行格式错误: {line!r}")
        i, j = int(parts[0]), int(parts[1])
        if i not in poses or j not in poses:
            raise TopoGraphError(f"edges_{edge_type}.txt 引用了不存在的关键帧: {i} / {j}")
        if use_file_weight:
            if len(parts) < 3:
                raise TopoGraphError(f"edges_{edge_type}.txt 缺权重列: {line!r}")
            weight = float(parts[2])
        else:
            weight = math.hypot(poses[i].x - poses[j].x, poses[i].y - poses[j].y)
        edges.append((i, j, weight))
    adjacency = _adjacency_from_weighted_edges(poses, edges)
    return TopoGraph(poses=poses, adjacency=adjacency)


def from_edges(poses: Dict[int, Pose2D], edges: Iterable[Tuple[int, int]]) -> TopoGraph:
    """从内存里的位姿 + 边列表建图,边长按两端机身位置的欧氏距离自动算。

    方便单测,也方便 LiteVLoc 侧直接从已有节点数据建图,不用先落盘成 ``poses.txt``。
    """
    weighted = (
        (i, j, math.hypot(poses[i].x - poses[j].x, poses[i].y - poses[j].y)) for i, j in edges
    )
    adjacency = _adjacency_from_weighted_edges(poses, weighted)
    return TopoGraph(poses=dict(poses), adjacency=adjacency)
