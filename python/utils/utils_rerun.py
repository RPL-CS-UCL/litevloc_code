#!/usr/bin/env python3
"""Rerun logging utilities for LiteVLoc offline visualization.

Entity path conventions
-----------------------
map/nodes/{id}              : per-node Transform3D (timeless)
map/nodes/{id}/camera       : per-node Pinhole + RGB Image (timeless)
map/nodes/{id}/body         : per-node Boxes3D green cube half_size=0.05 (timeless)
map/edges/covis             : covisibility edges LineStrips3D blue (timeless)
map/edges/trav              : traversability edges LineStrips3D blue (timeless)
world/axes                  : XYZ world frame axes Arrows3D red/green/blue (timeless)
query/pose_gt               : GT pose Transform3D (per-frame)
query/pose_gt/camera        : GT pose Pinhole + RGB Image (per-frame)
query/pose_estimated        : estimated pose Transform3D (per-frame)
query/pose_estimated/camera : estimated pose Pinhole + RGB Image (per-frame)
query/pose_estimated/axes   : XYZ pose axes Arrows3D red/green/blue (per-frame)
query/trajectory_gt         : accumulated GT trajectory LineStrips3D green (per-frame)
query/trajectory_est        : accumulated estimated trajectory LineStrips3D red (per-frame)
query/matching/{node_id}/combined : ref+query combined image with up to 20 match lines (per-frame)

Colors
------
GT pose / trajectory  : green (80, 220, 80)
Est pose / trajectory : red   (220, 80, 80)
Map edges             : blue  (100, 149, 237)
Map node body         : green box (0, 180, 100)
World / pose axes     : red X / green Y / blue Z (220,50,50)/(50,220,50)/(50,50,220)
Match keypoints lines : bright green (0, 220, 0)

Timeline
--------
Map entities: timeless=True.
Query/matching entities: frame_id / timestamp timelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rerun as rr


def init_rerun(app_id: str = "litevloc_offline_vloc") -> None:
    rr.init(app_id, spawn=False)


def save_rrd(output_path: Path) -> None:
    rr.save(str(output_path))


def set_frame_time(frame_id: int, timestamp: float) -> None:
    rr.set_time_sequence("frame_id", frame_id)
    rr.set_time_seconds("timestamp", timestamp)


def log_world_frame_axes(length: float = 1.0, radii: float = 0.01) -> None:
    """Log XYZ world frame axes as timeless Arrows3D (red=X, green=Y, blue=Z)."""
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
    """Log a pose as XYZ axes (red=X, green=Y, blue=Z) on the current frame timeline."""
    from scipy.spatial.transform import Rotation as R

    rot_mat = R.from_quat(quat).as_matrix()
    origin = trans.tolist()
    rr.log(
        entity_path,
        rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[
                (rot_mat @ np.array([length, 0.0, 0.0])).tolist(),
                (rot_mat @ np.array([0.0, length, 0.0])).tolist(),
                (rot_mat @ np.array([0.0, 0.0, length])).tolist(),
            ],
            radii=radii,
            colors=np.array([[220, 50, 50], [50, 220, 50], [50, 50, 220]], dtype=np.uint8),
        ),
    )


def log_map_nodes(graph) -> None:
    """Log all map nodes as timeless camera frustums with RGB images."""
    from scipy.spatial.transform import Rotation as R
    from utils.utils_image import to_numpy

    half_size = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    for node in graph.nodes.values():
        entity = f"map/nodes/{node.id}"
        width, height = int(node.img_size[0]), int(node.img_size[1])
        rot_mat = R.from_quat(node.quat).as_matrix()

        rr.log(entity, rr.Transform3D(translation=node.trans.tolist(), mat3x3=rot_mat.tolist()), timeless=True)
        rr.log(entity + "/camera", rr.Pinhole(image_from_camera=node.K, width=width, height=height), timeless=True)
        rr.log(entity + "/body", rr.Boxes3D(half_sizes=[half_size], colors=np.array([[0, 180, 100]], dtype=np.uint8)), timeless=True)
        if node.rgb_image is not None:
            rgb_np = (np.transpose(to_numpy(node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
            rr.log(entity + "/camera", rr.Image(rgb_np), timeless=True)


def log_map_edges(graph, edge_type: str = "covis") -> None:
    """Log map edges as timeless blue LineStrips3D."""
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
                radii=0.0025,
                colors=np.array([[100, 149, 237]], dtype=np.uint8),
            ),
            timeless=True,
        )


def log_query_camera(
    trans: np.ndarray,
    quat: np.ndarray,
    K: np.ndarray,
    img_size: Tuple[int, int],
    rgb_image: Optional[np.ndarray] = None,
    is_gt: bool = False,
) -> None:
    """Log a query camera frustum with optional RGB image.
    Orientation is conveyed by the Transform3D + Pinhole frustum; no separate arrow needed.
    """
    from scipy.spatial.transform import Rotation as R

    entity = "query/pose_gt" if is_gt else "query/pose_estimated"
    width, height = int(img_size[0]), int(img_size[1])
    rot_mat = R.from_quat(quat).as_matrix()

    rr.log(entity, rr.Transform3D(translation=trans.tolist(), mat3x3=rot_mat.tolist()))
    rr.log(entity + "/camera", rr.Pinhole(image_from_camera=K, width=width, height=height))
    if rgb_image is not None:
        rr.log(entity + "/camera", rr.Image(rgb_image))


def log_trajectory(
    positions: List[np.ndarray],
    is_gt: bool = False,
) -> None:
    """Log accumulated trajectory as LineStrips3D (GT=green, Est=red)."""
    if len(positions) < 2:
        return
    entity = "query/trajectory_gt" if is_gt else "query/trajectory_est"
    color = (80, 220, 80, 220) if is_gt else (220, 80, 80, 220)
    rr.log(
        entity,
        rr.LineStrips3D(
            strips=[np.array(positions, dtype=np.float32)],
            radii=0.01,
            colors=np.array([color], dtype=np.uint8),
        ),
    )


def log_image_matching(
    ref_image: np.ndarray,
    query_image: np.ndarray,
    mkpts_ref: np.ndarray,
    mkpts_query: np.ndarray,
    node_id: int,
    max_kpts: int = 20,
) -> None:
    """Log ref+query combined image with up to max_kpts green match lines."""
    import cv2

    h_ref, w_ref = ref_image.shape[:2]
    h_qry = query_image.shape[0]
    assert h_ref == h_qry, f"Height mismatch: ref={h_ref} vs query={h_qry}"

    combined = np.hstack([ref_image, query_image]).copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Reference", (8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "Query", (w_ref + 8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if len(mkpts_ref) > 0:
        idx = np.linspace(0, len(mkpts_ref) - 1, min(max_kpts, len(mkpts_ref)), dtype=int)
        for (x0, y0), (x1, y1) in zip(mkpts_ref[idx].astype(int), mkpts_query[idx].astype(int)):
            cv2.line(combined, (x0, y0), (x1 + w_ref, y1), (0, 220, 0), 1, cv2.LINE_AA)

    rr.log(f"query/matching/{node_id}/combined", rr.Image(combined))
