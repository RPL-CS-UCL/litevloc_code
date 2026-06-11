#!/usr/bin/env python3
"""Rerun logging utilities for LiteVLoc offline visualization.

Entity path conventions
-----------------------
map/nodes/{id}              : per-node camera frustum (Transform3D + Pinhole + Boxes3D), timeless
map/gallery/{id}            : per-node standalone RGB image for 2D gallery view, timeless
map/edges/covis             : covisibility edges (LineStrips3D), timeless
map/edges/trav              : traversability edges (LineStrips3D), timeless
query/image                 : current query RGB image, per-frame
query/pose_estimated        : estimated pose frustum + arrow, per-frame
query/pose_gt               : ground-truth pose frustum + arrow, per-frame
query/trajectory_est        : accumulated estimated trajectory (LineStrips3D), per-frame
query/trajectory_gt         : accumulated gt trajectory (LineStrips3D), per-frame
query/matching/{node_id}/ref_image   : matched reference map image
query/matching/{node_id}/query_image : query image with keypoints
query/matching/{node_id}/keypoints_ref   : keypoint overlay on reference (Points2D)
query/matching/{node_id}/keypoints_query : keypoint overlay on query (Points2D)

Timeline
--------
Map entities: explicit timeless=True → visible on all frames.
Query and matching entities use frame_id / timestamp timelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import rerun as rr


def init_rerun(app_id: str = "litevloc_offline_vloc") -> None:
    rr.init(app_id, spawn=False)


def save_rrd(output_path: Path) -> None:
    rr.save(str(output_path))


def log_map_nodes(graph) -> None:
    for node in graph.nodes.values():
        entity = f"map/nodes/{node.id}"
        _log_camera_frustum(entity, node.trans, node.quat, node.K, node.img_size, timeless=True)
        if node.rgb_image is not None:
            _log_image_on_frustum(entity + "/camera", node.rgb_image, timeless=True)


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
        rr.log(
            f"map/edges/{edge_type}",
            rr.LineStrips3D(
                strips=strips,
                radii=0.005,
                colors=np.array([[100, 149, 237]], dtype=np.uint8),
            ),
            timeless=True,
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
    _log_camera_frustum(entity, trans, quat, K, img_size)
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
    node_id: int,
) -> None:
    prefix = f"query/matching/{node_id}"
    rr.log(prefix + "/ref_image", rr.Image(ref_image))
    rr.log(prefix + "/query_image", rr.Image(query_image))

    if len(mkpts_ref) > 0:
        rr.log(
            prefix + "/keypoints_ref",
            rr.Points2D(
                positions=mkpts_ref, radii=3.0,
                colors=np.array([[0, 220, 0]], dtype=np.uint8),
            ),
        )
        rr.log(
            prefix + "/keypoints_query",
            rr.Points2D(
                positions=mkpts_query, radii=3.0,
                colors=np.array([[0, 220, 0]], dtype=np.uint8),
            ),
        )


def _log_camera_frustum(
    entity_path: str,
    trans: np.ndarray,
    quat: np.ndarray,
    K: np.ndarray,
    img_size: np.ndarray,
    timeless: bool = False,
) -> None:
    from scipy.spatial.transform import Rotation as R

    width, height = int(img_size[0]), int(img_size[1])
    rot_mat = R.from_quat(quat).as_matrix()
    half_size = np.array([0.04, 0.04, 0.04], dtype=np.float32)

    rr.log(
        entity_path,
        rr.Transform3D(translation=trans.tolist(), mat3x3=rot_mat.tolist()),
        timeless=timeless,
    )
    rr.log(
        entity_path + "/camera",
        rr.Pinhole(image_from_camera=K, width=width, height=height),
        timeless=timeless,
    )


def _log_image_on_frustum(entity_path: str, rgb_tensor, timeless: bool = False) -> None:
    from utils.utils_image import to_numpy
    rgb_np = (np.transpose(to_numpy(rgb_tensor), (1, 2, 0)) * 255).astype(np.uint8)
    rr.log(entity_path, rr.Image(rgb_np), timeless=timeless)


def _log_map_node_gallery_image(entity_path: str, rgb_tensor) -> None:
    from utils.utils_image import to_numpy

    rgb_np = (np.transpose(to_numpy(rgb_tensor), (1, 2, 0)) * 255).astype(np.uint8)
    rr.log(entity_path, rr.Image(rgb_np), timeless=True)


def _log_pose_arrow(
    entity_path: str,
    trans: np.ndarray,
    quat: np.ndarray,
    color: Tuple[int, int, int, int],
    length: float = 0.15,
) -> None:
    from scipy.spatial.transform import Rotation as R

    rot_mat = R.from_quat(quat).as_matrix()
    forward = rot_mat @ np.array([0.0, 0.0, length])
    rr.log(
        entity_path,
        rr.Arrows3D(
            origins=[trans.tolist()],
            vectors=[forward.tolist()],
            radii=0.015,
            colors=np.array([color], dtype=np.uint8),
        ),
    )
