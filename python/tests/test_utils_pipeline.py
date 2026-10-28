"""Unit tests for utils/utils_pipeline.py's matcher_profiles.yaml config-filling (M5-B4).

No ROS / torch / GPU needed. parse_arguments() itself imports vismatch (for `available_models`)
and utils.pose_solver (for `available_solvers`), both pure-Python/OpenCV, so this only needs
LITEVLOC_PY and vismatch on sys.path.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

LITEVLOC_PY = pathlib.Path(__file__).resolve().parents[1]
VISMATCH = LITEVLOC_PY.parents[1] / "vismatch"
for _p in (LITEVLOC_PY, VISMATCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils_pipeline import (  # noqa: E402
    fill_args_from_matcher_profile,
    load_matcher_profile,
    parse_arguments,
)


@pytest.fixture()
def tmp_profiles(tmp_path) -> str:
    """A small matcher_profiles.yaml with one matcher, used to test the config-fill logic
    without depending on the real config/matcher_profiles.yaml (which may change its placeholder
    gv_min_inliers numbers once the main session fills them in)."""
    path = tmp_path / "matcher_profiles.yaml"
    path.write_text(
        yaml.safe_dump({
            "edm": {
                "image_size": [512, 384],
                "n_kpts": 2048,
                "gv_min_inliers": 30,
                "gv_min_inliers_nodepth": 30,
                "matcher_kwargs": {"ransac_model": "none"},
            },
            "master": {
                "image_size": [512, 384],
                "n_kpts": 2048,
                "gv_min_inliers": 100,
                "gv_min_inliers_nodepth": 100,
                "matcher_kwargs": {"min_conf_thr": 1.5},
            },
        })
    )
    return str(path)


def _parse(argv, tmp_profiles):
    old_argv = sys.argv
    sys.argv = ["prog"] + argv + ["--matcher_config", tmp_profiles]
    try:
        return parse_arguments()
    finally:
        sys.argv = old_argv


def test_load_matcher_profile_known_and_unknown(tmp_profiles):
    assert load_matcher_profile("edm", tmp_profiles)["gv_min_inliers"] == 30
    assert load_matcher_profile("totally-unknown-matcher", tmp_profiles) == {}


def test_fill_from_profile_when_not_given_on_cli(tmp_profiles):
    args = _parse(["--img_matcher", "edm"], tmp_profiles)
    assert args.image_size == [512, 384]
    assert args.n_kpts == 2048
    assert args.gv_min_inliers == 30
    assert args.gv_min_inliers_nodepth == 30
    assert args.matcher_kwargs == {"ransac_model": "none"}


def test_explicit_cli_args_win_over_profile(tmp_profiles):
    """命令行显式给的参数优先于配置文件里的值。"""
    args = _parse(
        [
            "--img_matcher", "edm",
            "--image_size", "640", "480",
            "--n_kpts", "999",
            "--gv_min_inliers", "7",
            "--gv_min_inliers_nodepth", "8",
        ],
        tmp_profiles,
    )
    assert args.image_size == [640, 480]
    assert args.n_kpts == 999
    assert args.gv_min_inliers == 7
    assert args.gv_min_inliers_nodepth == 8
    # matcher_kwargs 不受这些参数影响,仍然来自配置(edm 没有 min_master_conf_thre 这类重叠项)
    assert args.matcher_kwargs == {"ransac_model": "none"}


def test_explicit_min_master_conf_thre_overrides_profile_min_conf_thr(tmp_profiles):
    args = _parse(["--img_matcher", "master", "--min_master_conf_thre", "9.9"], tmp_profiles)
    assert args.matcher_kwargs["min_conf_thr"] == 9.9
    assert args.min_master_conf_thre == 9.9


def test_profile_min_conf_thr_used_when_not_given_on_cli(tmp_profiles):
    args = _parse(["--img_matcher", "master"], tmp_profiles)
    assert args.matcher_kwargs["min_conf_thr"] == 1.5
    assert args.min_master_conf_thre == 1.5


def test_unknown_matcher_falls_back_to_defaults_without_crashing(tmp_profiles):
    args = _parse(["--img_matcher", "xfeat"], tmp_profiles)  # not in tmp_profiles
    assert args.n_kpts == 2048  # hardcoded fallback, not from any profile
    assert args.gv_min_inliers == 100.0  # GV_SCORE_THRESHOLD fallback
    assert args.matcher_kwargs == {}


def test_fill_args_from_matcher_profile_is_idempotent_on_prefilled_args():
    """fill_args_from_matcher_profile 直接拿一个已经填好值的 args 再调一次,不应该覆盖已有值
    (getattr(..., None) is None 才填的保护逻辑)。"""
    import argparse

    args = argparse.Namespace(
        img_matcher="edm", image_size=[1, 1], n_kpts=3,
        gv_min_inliers=4, gv_min_inliers_nodepth=5, min_master_conf_thre=6.0,
    )
    fill_args_from_matcher_profile(args, config_path="/nonexistent/path.yaml")
    assert args.image_size == [1, 1]
    assert args.n_kpts == 3
    assert args.gv_min_inliers == 4
    assert args.gv_min_inliers_nodepth == 5
    assert args.matcher_kwargs == {"min_conf_thr": 6.0}


def test_real_profile_edm_requires_more_than_200_inliers():
    """用户 10-28 定:EDM 定位的核验内点 > 200 才算成功(全局定位和本地跟踪共用这个门槛)。"""
    from utils.utils_pipeline import load_matcher_profile

    profile = load_matcher_profile("edm")
    assert profile["gv_min_inliers"] == 201
    assert 200 < profile["gv_min_inliers"]  # 200 不算,201 算
