#!/usr/bin/env python3
"""Rerun logging utilities for LiteVLoc offline visualization.

Entity path conventions
-----------------------
map/nodes/{id}              : per-node camera frustum (Transform3D + Pinhole), timeless
map/nodes/{id}/camera       : per-node RGB image on Pinhole frustum, timeless
map/edges/covis             : covisibility edges (LineStrips3D), timeless
map/edges/trav              : traversability edges (LineStrips3D), timeless
world/axes                  : XYZ world frame axes (red/green/blue Arrows3D), timeless
query/image                 : current query RGB image, per-frame
query/pose_estimated        : estimated pose frustum + arrow, per-frame
query/pose_estimated/axes   : XYZ pose axes for estimated pose (red/green/blue Arrows3D), per-frame
query/pose_gt               : ground-truth pose frustum + arrow, per-frame
query/trajectory_est        : accumulated estimated trajectory (LineStrips3D), per-frame
query/trajectory_gt         : accumulated gt trajectory (LineStrips3D), per-frame
query/matching/{node_id}/combined    : ref + query horizontally combined with match lines, per-frame

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


def log_world_frame_axes(length: float = 0.5, radii: float = 0.02) -> None:
    origin = [0.0, 0.0, 0.0]
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[[length, 0, 0], [0, length, 0], [0, 0, length]],
            radii=radii,
            colors=np.array([[220, 50, 50], [50, 220, 50], [50, 50, 220]], dtype=np.uint8),
        ),
        timeless=True,
    )


def log_pose_axes(
    trans: np.ndarray,
    quat: np.ndarray,
    entity_path: str = "query/pose_estimated/axes",
    length: float = 0.15,
    radii: float = 0.015,
) -> None:
    from scipy.spatial.transform import Rotation as R

    rot_mat = R.from_quat(quat).as_matrix()
    x_axis = rot_mat @ np.array([length, 0.0, 0.0])
    y_axis = rot_mat @ np.array([0.0, length, 0.0])
    z_axis = rot_mat @ np.array([0.0, 0.0, length])
    origin = trans.tolist()
    rr.log(
        entity_path,
        rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[x_axis.tolist(), y_axis.tolist(), z_axis.tolist()],
            radii=radii,
            colors=np.array([[220, 50, 50], [50, 220, 50], [50, 50, 220]], dtype=np.uint8),
        ),
    )


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
    color = (80, 220, 80, 220) if is_gt else (220, 80, 80, 220)
    _log_camera_frustum(entity, trans, quat, K, img_size)
    _log_pose_arrow(entity + "/arrow", trans, quat, color)


def log_trajectory(
    positions: List[np.ndarray],
    is_gt: bool = False,
) -> None:
    if len(positions) < 2:
        return
    entity = "query/trajectory_gt" if is_gt else "query/trajectory_est"
    color = (80, 220, 80, 220) if is_gt else (220, 80, 80, 220)
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
    import cv2

    h_ref, w_ref = ref_image.shape[:2]
    h_qry, w_qry = query_image.shape[:2]

    assert h_ref == h_qry, f"Height mismatch: ref={h_ref} vs query={h_qry}"

    combined = np.hstack([ref_image, query_image]).copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Reference", (8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "Query", (w_ref + 8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if len(mkpts_ref) > 0:
        for (x0, y0), (x1, y1) in zip(mkpts_ref.astype(int), mkpts_query.astype(int)):
            cv2.line(combined, (x0, y0), (x1 + w_ref, y1), (0, 220, 0), 1, cv2.LINE_AA)

    rr.log(f"query/matching/{node_id}/combined", rr.Image(combined))


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
