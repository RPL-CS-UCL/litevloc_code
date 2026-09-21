"""Align a local metric trajectory to GPS: ENU (first valid fix as origin) + Kabsch.

Extracted from utils_convert_pose_to_kml.py so the KML script and other callers
(e.g. the OpenNavMap console) share one implementation. Depends on numpy + pymap3d only.
"""
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pymap3d as pm

Geodetic = Tuple[float, float, float]  # (lat [deg], lon [deg], alt [m])
GpsPair = Tuple[np.ndarray, Geodetic]  # (local xyz, geodetic)


def collect_gps_pairs(positions: Sequence[np.ndarray],
                      gps_rows: Sequence[Optional[Sequence[float]]]) -> Tuple[List[GpsPair], Optional[Geodetic]]:
    """Pair each local position with its GPS row ([lat, lon, alt, ...] or None).

    Rows with a nan lat/lon are skipped; a nan altitude counts as 0. The origin is the
    first valid fix (None when there is none).
    """
    pairs: List[GpsPair] = []
    origin: Optional[Geodetic] = None
    for pos, gps_data in zip(positions, gps_rows):
        if gps_data is None:
            continue
        lat, lon = gps_data[:2]
        alt = gps_data[2] if len(gps_data) > 2 else 0.0
        if math.isnan(alt):
            alt = 0.0
        if any(math.isnan(v) for v in (lat, lon, alt)):
            continue
        if origin is None:
            origin = (lat, lon, alt)
        pairs.append((np.asarray(pos, dtype=float), (lat, lon, alt)))
    return pairs, origin


def compute_local_to_enu(pairs: Sequence[GpsPair], origin: Geodetic) -> np.ndarray:
    """4x4 rigid transform (Kabsch, no scale) mapping local xyz onto ENU about origin.

    Identity when fewer than two pairs are available.
    """
    T_ini = np.eye(4)
    if len(pairs) >= 2:
        enu_points, local_points = [], []
        for pose, gps in pairs:
            e, n, u = pm.geodetic2enu(gps[0], gps[1], gps[2], origin[0], origin[1], origin[2])
            enu_points.append([e, n, u])
            local_points.append(pose[:3] if len(pose) >= 3 else [0, 0, 0])

        enu_arr = np.array(enu_points)
        local_arr = np.array(local_points)

        # Kabsch algorithm to find optimal rotation and translation
        centroid_enu = np.mean(enu_arr, axis=0)
        centroid_local = np.mean(local_arr, axis=0)
        centered_enu = enu_arr - centroid_enu
        centered_local = local_arr - centroid_local

        H = centered_local.T @ centered_enu
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_enu - R @ centroid_local
        T_ini[:3, :3] = R
        T_ini[:3, 3] = t
    return T_ini


def local_to_geodetic(T_local_to_enu: np.ndarray, positions: Sequence[np.ndarray],
                      origin: Geodetic) -> np.ndarray:
    """Apply the transform and convert each local position to (lat, lon, alt); shape (N, 3)."""
    out = np.zeros((len(positions), 3))
    for i, pos in enumerate(positions):
        tx, ty, tz = (T_local_to_enu[:3, :3] @ np.asarray(pos, dtype=float) + T_local_to_enu[:3, 3])
        out[i] = pm.enu2geodetic(tx, ty, tz, origin[0], origin[1], origin[2])
    return out
