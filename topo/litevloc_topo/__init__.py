"""litevloc_topo:关键帧拓扑图上找最短路、算路点、管理目标点的纯 numpy 小包。

nav_mapping(nav_system)和 LiteVLoc(本仓库 ``global_planner.py`` / ``utils_shortest_path.py``)
共用这一份实现,不再各自维护一套 Dijkstra。用法见同目录 ``README.md``。
"""

from .goals import Goal, ImageGoal, KeyframeGoal, LanguageGoal, ObjectGoal, PoseGoal, TextResolver
from .graph import (
    Pose2D,
    TopoGraph,
    TopoGraphError,
    camera_to_body,
    from_edges,
    load_topo_graph,
    read_camera_poses,
    resolve_place_dir,
)
from .planner import TopoPlanner
from .route import (
    MAX_SEGMENT_M,
    MIN_WAYPOINT_SPACING_M,
    Route,
    path_to_waypoints,
    path_to_waypoints_with_kf,
    sparsify,
    subdivide,
)
from .search import NoPathError, dijkstra, k_hop

__all__ = [
    "Goal",
    "ImageGoal",
    "KeyframeGoal",
    "LanguageGoal",
    "ObjectGoal",
    "PoseGoal",
    "TextResolver",
    "Pose2D",
    "TopoGraph",
    "TopoGraphError",
    "camera_to_body",
    "from_edges",
    "load_topo_graph",
    "read_camera_poses",
    "resolve_place_dir",
    "TopoPlanner",
    "MAX_SEGMENT_M",
    "MIN_WAYPOINT_SPACING_M",
    "Route",
    "path_to_waypoints",
    "path_to_waypoints_with_kf",
    "sparsify",
    "subdivide",
    "NoPathError",
    "dijkstra",
    "k_hop",
]
