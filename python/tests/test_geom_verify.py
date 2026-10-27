"""Unit tests for utils/geom_verify.py (M5-B4 统一核验).

Pure geometry + OpenCV, no ROS / torch / GPU / real matcher needed: a pinhole camera
model, synthetic 3D points, and a pure-translation camera pair, projected by hand.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
import pytest

LITEVLOC_PY = pathlib.Path(__file__).resolve().parents[1]
if str(LITEVLOC_PY) not in sys.path:
    sys.path.insert(0, str(LITEVLOC_PY))

from utils import geom_verify  # noqa: E402
from utils.pose_solver import get_solver  # noqa: E402
from rpe_default import cfg as _CFG  # noqa: E402


def _pnp_cfg():
    """A fresh PNP-only yacs config (matches config/dataset/matterport3d.yaml's PNP block)."""
    cfg = _CFG.clone()
    cfg.PNP.RANSAC_ITER = 1000
    cfg.PNP.REPROJECTION_INLIER_THRESHOLD = 3
    cfg.PNP.CONFIDENCE = 0.9999
    return cfg


def _make_solver():
    return get_solver("pnp", _pnp_cfg())


def _project(K: np.ndarray, T_world_cam: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """Project world points into a camera's image plane given its world->... camera-to-world pose."""
    T_cam_world = np.linalg.inv(T_world_cam)
    pts_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
    pts_cam = (T_cam_world @ pts_h.T).T[:, :3]
    assert np.all(pts_cam[:, 2] > 0), "test points must be in front of the camera"
    uv_h = (K @ pts_cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    return uv, pts_cam[:, 2]


def _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.0, 0.0), seed=0):
    """Map camera at the world origin (identity pose); query camera translated by `translation`
    with the same orientation (pure translation, no rotation). Returns a dict with K, both camera
    poses, projected keypoints in both images, and a synthetic depth map for the query camera."""
    rng = np.random.default_rng(seed)
    K = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    img_w, img_h = 640, 480

    T_world_map = np.eye(4)
    T_world_query = np.eye(4)
    T_world_query[:3, 3] = np.array(translation, dtype=np.float64)

    # Random 3D points in front of both cameras (z in [2, 6] m in the map camera frame, since
    # that camera sits at the world origin with identity orientation).
    pts_world = np.column_stack([
        rng.uniform(-1.5, 1.5, n_pts),
        rng.uniform(-1.0, 1.0, n_pts),
        rng.uniform(2.0, 6.0, n_pts),
    ])

    kpts_map, _ = _project(K, T_world_map, pts_world)
    kpts_query, depth_query_vals = _project(K, T_world_query, pts_world)

    # Keep only points that land inside both images (a real matcher would never return a
    # correspondence pair where one side is off-frame).
    def _inside(uv):
        return (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)

    keep = _inside(kpts_map) & _inside(kpts_query)
    kpts_map, kpts_query, depth_query_vals = kpts_map[keep], kpts_query[keep], depth_query_vals[keep]
    assert len(kpts_map) >= min(50, n_pts), "test setup should keep most synthetic points in-frame"

    # Build a sparse-but-dense-enough depth image: nearest-pixel splat is fine since
    # verify_pnp/PnPSolver reads depth at np.int32(kpts_query) pixel locations.
    depth_query = np.zeros((img_h, img_w), dtype=np.float64)
    px = np.clip(kpts_query[:, 0].astype(np.int32), 0, img_w - 1)
    py = np.clip(kpts_query[:, 1].astype(np.int32), 0, img_h - 1)
    depth_query[py, px] = depth_query_vals

    return dict(
        K=K, T_world_map=T_world_map, T_world_query=T_world_query,
        kpts_map=kpts_map, kpts_query=kpts_query, depth_query=depth_query,
    )


# ---------------------------------------------------------------------------
# grid_dedup
# ---------------------------------------------------------------------------

def test_grid_dedup_keeps_one_point_per_cell():
    # Two points in the same 16px cell, one point far away in a different cell.
    kpts_query = np.array([[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]])
    kpts_map = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

    kq, km = geom_verify.grid_dedup(kpts_query, kpts_map, cell_px=16)

    assert kq.shape[0] == 2  # one per cell
    assert km.shape[0] == 2
    # The far-away point (different cell) must have survived.
    assert np.any(np.all(np.isclose(kq, [100.0, 100.0]), axis=1))


def test_grid_dedup_respects_max_points():
    n = 50
    # Spread points far apart so every one lands in its own cell.
    kpts_query = np.column_stack([np.arange(n) * 100.0, np.zeros(n)])
    kpts_map = kpts_query.copy()

    kq, km = geom_verify.grid_dedup(kpts_query, kpts_map, cell_px=16, max_points=10)

    assert kq.shape[0] == 10
    assert km.shape[0] == 10


def test_grid_dedup_empty_input():
    kq, km = geom_verify.grid_dedup(np.empty((0, 2)), np.empty((0, 2)))
    assert kq.shape == (0, 2)
    assert km.shape == (0, 2)


def test_grid_dedup_picks_highest_score_in_cell():
    kpts_query = np.array([[1.0, 1.0], [2.0, 2.0]])  # same cell
    kpts_map = np.array([[10.0, 10.0], [20.0, 20.0]])
    scores = np.array([0.1, 0.9])

    kq, km = geom_verify.grid_dedup(kpts_query, kpts_map, cell_px=16, scores=scores)

    assert kq.shape[0] == 1
    np.testing.assert_allclose(kq[0], [2.0, 2.0])
    np.testing.assert_allclose(km[0], [20.0, 20.0])


# ---------------------------------------------------------------------------
# verify_pnp: pure translation, synthetic depth
# ---------------------------------------------------------------------------

def test_verify_pnp_pure_translation_recovers_pose_and_inliers():
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.1, 0.0))
    solver = _make_solver()

    result = geom_verify.verify_pnp(
        data["kpts_query"], data["kpts_map"], data["depth_query"],
        data["K"], data["K"], data["T_world_map"], solver,
    )

    assert result.T_world_query is not None
    assert result.inliers >= 50
    t_err = np.linalg.norm(result.T_world_query[:3, 3] - data["T_world_query"][:3, 3])
    assert t_err < 0.01, f"translation error {t_err:.5f} m should be < 1cm"
    R_err = result.T_world_query[:3, :3] @ data["T_world_query"][:3, :3].T
    rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)))
    assert rot_err_deg < 1.0


def test_verify_pnp_grid_dedup_caps_points_used():
    """With a tiny cell size that collapses everything into very few cells, the inlier count
    must not exceed what grid_dedup actually kept (sanity: dedup really runs before PnP)."""
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.0, 0.0))
    solver = _make_solver()

    result = geom_verify.verify_pnp(
        data["kpts_query"], data["kpts_map"], data["depth_query"],
        data["K"], data["K"], data["T_world_map"], solver,
        cell_px=16, max_points=5,
    )
    assert result.inliers <= 5


def test_verify_pnp_too_few_points_after_dedup():
    data = _synthetic_pure_translation(n_pts=3, translation=(0.5, 0.0, 0.0))
    solver = _make_solver()
    result = geom_verify.verify_pnp(
        data["kpts_query"][:3], data["kpts_map"][:3], data["depth_query"],
        data["K"], data["K"], data["T_world_map"], solver,
    )
    assert result.inliers == 0
    assert result.T_world_query is None


def test_verify_pnp_outlier_matches_excluded_from_inlier_count():
    """Add ~30% random (wrong) correspondences on top of the good ones; the PnP-RANSAC inlier
    count must reflect only the geometrically-consistent (good) matches, not the outliers."""
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.1, 0.0))
    solver = _make_solver()
    n_good = data["kpts_query"].shape[0]

    rng = np.random.default_rng(42)
    n_bad = max(1, int(0.3 * n_good))
    bad_query = np.column_stack([rng.uniform(0, 640, n_bad), rng.uniform(0, 480, n_bad)])
    bad_map = np.column_stack([rng.uniform(0, 640, n_bad), rng.uniform(0, 480, n_bad)])

    kpts_query = np.vstack([data["kpts_query"], bad_query])
    kpts_map = np.vstack([data["kpts_map"], bad_map])
    # Random depth values for the fake outlier points (plausible range).
    depth_query = data["depth_query"].copy()

    result = geom_verify.verify_pnp(
        kpts_query, kpts_map, depth_query,
        data["K"], data["K"], data["T_world_map"], solver,
        cell_px=1, max_points=10000,  # effectively disable dedup so all points reach PnP
    )

    assert result.T_world_query is not None
    # inliers must come only from the up-to-n_good good correspondences, never count the outliers
    assert result.inliers <= n_good
    t_err = np.linalg.norm(result.T_world_query[:3, 3] - data["T_world_query"][:3, 3])
    assert t_err < 0.05


# ---------------------------------------------------------------------------
# verify_fundamental: no-depth fallback
# ---------------------------------------------------------------------------

def test_verify_fundamental_consistent_matches_pass():
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.1, 0.05))
    inliers = geom_verify.verify_fundamental(data["kpts_query"], data["kpts_map"])
    assert inliers >= 50


def test_verify_fundamental_too_few_points():
    data = _synthetic_pure_translation(n_pts=5, translation=(0.5, 0.0, 0.0))
    inliers = geom_verify.verify_fundamental(data["kpts_query"][:5], data["kpts_map"][:5])
    assert inliers == 0


def test_verify_fundamental_opencv_error_returns_zero(monkeypatch):
    """Some point sets make ``cv2.findFundamentalMat`` raise ``cv2.error`` instead of returning
    an empty mask (seen in practice: ``!model.empty() in function 'setModelParameters'``). That
    must be treated the same as "could not solve" (inlier count 0), not propagate up and crash
    the caller."""
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.1, 0.05))

    def _raise(*_args, **_kwargs):
        raise cv2.error("(-215:Assertion failed) !model.empty() in function 'setModelParameters'")

    monkeypatch.setattr(geom_verify.cv2, "findFundamentalMat", _raise)
    inliers = geom_verify.verify_fundamental(data["kpts_query"], data["kpts_map"])
    assert inliers == 0


# ---------------------------------------------------------------------------
# verify_match_result: shared by LocPipeline (ROS) and run_vloc_offline_rerun.py (no ROS)
# ---------------------------------------------------------------------------

def test_verify_match_result_rescales_and_uses_pnp_with_depth():
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.1, 0.0))
    solver = _make_solver()

    # Pretend both images were resized to half resolution before matching: match_result's
    # kpts are in that half-resolution frame, raw_img_size is the original 640x480.
    scale = 0.5
    match_result = {
        "inlier_kpts0": data["kpts_map"] * scale,  # map image
        "inlier_kpts1": data["kpts_query"] * scale,  # query image
    }

    result = geom_verify.verify_match_result(
        match_result,
        map_raw_K=data["K"], map_raw_img_size=(640, 480), map_img_size=(320, 240),
        query_raw_K=data["K"], query_raw_img_size=(640, 480), query_img_size=(320, 240),
        T_world_map_cam=data["T_world_map"],
        depth_query=data["depth_query"],
        solver=solver,
    )

    assert result.T_world_query is not None
    assert result.inliers >= 50
    t_err = np.linalg.norm(result.T_world_query[:3, 3] - data["T_world_query"][:3, 3])
    assert t_err < 0.01


def test_verify_match_result_without_depth_falls_back_to_fundamental():
    data = _synthetic_pure_translation(n_pts=200, translation=(0.5, 0.1, 0.05))
    solver = _make_solver()
    match_result = {"inlier_kpts0": data["kpts_map"], "inlier_kpts1": data["kpts_query"]}

    result = geom_verify.verify_match_result(
        match_result,
        map_raw_K=data["K"], map_raw_img_size=(640, 480), map_img_size=(640, 480),
        query_raw_K=data["K"], query_raw_img_size=(640, 480), query_img_size=(640, 480),
        T_world_map_cam=data["T_world_map"],
        depth_query=None,
        solver=solver,
    )

    assert result.T_world_query is None
    assert result.inliers >= 50
