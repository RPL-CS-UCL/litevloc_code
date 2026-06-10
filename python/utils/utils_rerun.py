#!/usr/bin/env python3
"""Rerun logging utilities for LiteVLoc offline visualization.

Entity path conventions
-----------------------
map/nodes/points           : all map node positions (Points3D)
map/nodes/cameras/{id}     : per-node camera frustum (Pinhole + Transform3D)
map/edges/covis            : covisibility edges (LineStrips3D)
map/edges/trav             : traversability edges (LineStrips3D)
query/camera               : current query camera frustum
query/image                : current query RGB image
query/pose_estimated       : estimated pose arrow (Arrows3D)
query/pose_gt              : ground-truth pose arrow (Arrows3D)
query/trajectory_est       : accumulated estimated trajectory (LineStrips3D)
query/trajectory_gt        : accumulated gt trajectory (LineStrips3D)
matching/ref_image         : matched reference map image
matching/query_image       : query image with keypoints
matching/keypoints_ref     : keypoint overlay on reference (Points2D)
matching/keypoints_query   : keypoint overlay on query (Points2D)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import rerun as rr


def init_rerun(app_id: str = "litevloc_offline") -> None:
    rr.init(app_id)


def save_rrd(output_path: Path) -> None:
    rr.save(str(output_path))


def log_map_nodes(graph) -> None:
    positions = []
    for node in graph.nodes.values():
        positions.append(node.trans)

    if positions:
        rr.log(
            "map/nodes/points",
            rr.Points3D(
                positions=np.array(positions, dtype=np.float32),
                radii=0.05,
            ),
        )

    for node in graph.nodes.values():
        _log_camera_frustum(
            entity_path=f"map/nodes/cameras/{node.id}",
            trans=node.trans,
            quat=node.quat,
            K=node.raw_K,
            img_size=node.raw_img_size,
            color=(0, 180, 100, 180),
        )


def log_map_edges(graph, edge_type: str = "covis") -> None:
    strips = []
    visited = set()
    for node in graph.nodes.values():
        for neighbor, _ in node.edges.values():
            pair = (min(node.id, neighbor.id), max(node.id, neighbor.id))
            if pair in visited:
                continue
            visited.add(pair)
            strips.append(np.array([node.trans, neighbor.trans], dtype=np.float32))

    if strips:
        color = (0, 120, 200, 180) if edge_type == "covis" else (200, 130, 0, 180)
        rr.log(
            f"map/edges/{edge_type}",
            rr.LineStrips3D(strips=strips, colors=np.array([color], dtype=np.uint8)),
        )


def set_frame_time(frame_id: int, timestamp: float) -> None:
    rr.set_time_sequence("frame_id", frame_id)
    rr.set_time_seconds("timestamp", timestamp)


def log_query_image(rgb_image: np.ndarray) -> None:
    rr.log("query/image", rr.Image(rgb_image))


def log_query_camera(
    trans: np.ndarray,
    quat: np.ndarray,
    K: np.ndarray,
    img_size: Tuple[int, int],
    is_gt: bool = False,
) -> None:
    entity = "query/pose_gt" if is_gt else "query/pose_estimated"
    color = (255, 80, 80, 220) if is_gt else (80, 80, 255, 220)
    _log_camera_frustum(entity, trans, quat, K, img_size, color)
    _log_pose_arrow(entity + "/arrow", trans, quat, color)


def log_trajectory(
    positions: List[np.ndarray],
    is_gt: bool = False,
) -> None:
    if len(positions) < 2:
        return
    entity = "query/trajectory_gt" if is_gt else "query/trajectory_est"
    color = (255, 80, 80, 220) if is_gt else (80, 80, 255, 220)
    rr.log(
        entity,
        rr.LineStrips3D(
            strips=[np.array(positions, dtype=np.float32)],
            colors=np.array([color], dtype=np.uint8),
        ),
    )


def log_image_matching(
    ref_image: np.ndarray,
    query_image: np.ndarray,
    mkpts_ref: np.ndarray,
    mkpts_query: np.ndarray,
) -> None:
    rr.log("matching/ref_image", rr.Image(ref_image))
    rr.log("matching/query_image", rr.Image(query_image))

    if len(mkpts_ref) > 0:
        rr.log(
            "matching/keypoints_ref",
            rr.Points2D(
                positions=mkpts_ref,
                radii=3.0,
                colors=np.array([[0, 220, 0]], dtype=np.uint8),
            ),
        )
        rr.log(
            "matching/keypoints_query",
            rr.Points2D(
                positions=mkpts_query,
                radii=3.0,
                colors=np.array([[0, 220, 0]], dtype=np.uint8),
            ),
        )


def _log_camera_frustum(
    entity_path: str,
    trans: np.ndarray,
    quat: np.ndarray,
    K: np.ndarray,
    img_size: np.ndarray,
    color: Tuple[int, int, int, int],
) -> None:
    from scipy.spatial.transform import Rotation as R

    width, height = int(img_size[0]), int(img_size[1])
    rot_mat = R.from_quat(quat).as_matrix()

    rr.log(
        entity_path,
        rr.Transform3D(
            translation=trans.tolist(),
            mat3x3=rot_mat.tolist(),
        ),
    )
    rr.log(
        entity_path + "/camera",
        rr.Pinhole(
            image_from_camera=K,
            width=width,
            height=height,
        ),
    )


def _log_pose_arrow(
    entity_path: str,
    trans: np.ndarray,
    quat: np.ndarray,
    color: Tuple[int, int, int, int],
    length: float = 0.5,
) -> None:
    from scipy.spatial.transform import Rotation as R

    rot_mat = R.from_quat(quat).as_matrix()
    forward = rot_mat @ np.array([0.0, 0.0, length])
    rr.log(
        entity_path,
        rr.Arrows3D(
            origins=[trans.tolist()],
            vectors=[forward.tolist()],
            colors=np.array([color], dtype=np.uint8),
        ),
    )
