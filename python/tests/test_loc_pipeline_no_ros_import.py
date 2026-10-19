"""M5-B4b: `loc_pipeline.py` 的 ROS import(rospy / tf2_ros / 消息类型 / utils.utils_ros)从模块
顶部挪进了真正用到它们的方法里，这样 `run_vloc_offline_rerun.py`（它的运行环境——见
`robots/go2/runtime/start_vloc_offline.sh`——故意把 ROS 从 PYTHONPATH 里剔掉了）才能
`import loc_pipeline` 并调 `LocPipeline.localize_obs`。

这个文件证明两件事，跟 `tests/test_ros_loc_pipeline.py` 里已经有的 localize_obs 正确性测试
(用 MagicMock 假装 ROS 存在)不是一回事：

1. ROS 真的不可用时(不是"还没 import 过"，是 sys.modules 里显式写了 None——Python 的 import
   机制把这个当"已确认找不到"，比 monkeypatch.delitem 更贴近 start_vloc_offline.sh 的真实场景)
   `import loc_pipeline` 仍然成功。
2. 这种环境下 `LocPipeline.localize_obs`(用一个不碰 ROS 的替身 img_matcher)能正常跑完一次
   global 核验和一次 local 核验，不触碰任何 ROS 符号。
3. 顺手验证 import 确实被推迟了，不是"反正没人用到 ROS 所以测不出差异"：调用一个在线专用方法
   (`init_vpr_model`，内部会 `import rospy`)在这种环境下必须因为 rospy 不可用而报错。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

LITEVLOC_PY = Path(__file__).resolve().parents[1]

# sys.modules[name] = None 是 Python import 机制认可的"确认找不到"标记，比直接不存在这个 key
# 更强——哪怕系统里装着真的 rospy，这样也会让 `import rospy` 照样报 ImportError。
_UNAVAILABLE_ROS_MODULES = [
    "rospy", "tf2_ros",
    "std_msgs", "std_msgs.msg",
    "nav_msgs", "nav_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg",
    "geometry_msgs", "geometry_msgs.msg",
    "visualization_msgs", "visualization_msgs.msg",
    "utils.utils_ros",
]


@pytest.fixture()
def loc_pipeline_mod_no_ros(monkeypatch):
    monkeypatch.syspath_prepend(str(LITEVLOC_PY))
    for name in _UNAVAILABLE_ROS_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "loc_pipeline", raising=False)
    import loc_pipeline as module
    yield module
    monkeypatch.delitem(sys.modules, "loc_pipeline", raising=False)


def test_import_loc_pipeline_without_ros(loc_pipeline_mod_no_ros):
    """`import loc_pipeline` 在 rospy/tf2_ros/消息包都确认不可用时必须成功。"""
    assert hasattr(loc_pipeline_mod_no_ros, "LocPipeline")
    assert hasattr(loc_pipeline_mod_no_ros, "LocResult")
    # 真的没有把 rospy 换成别的什么东西悄悄塞进 sys.modules(还是 None,证明没人 import 成功过)。
    assert sys.modules.get("rospy") is None


def test_online_only_method_fails_without_ros(loc_pipeline_mod_no_ros):
    """反向验证:import 能成功不是因为代码根本不需要 ROS,而是推迟到了方法调用时才需要——
    `init_vpr_model`(在线节点用,内部 `import rospy`)在这种环境下必须报错。"""
    pipeline = loc_pipeline_mod_no_ros.LocPipeline.__new__(loc_pipeline_mod_no_ros.LocPipeline)
    pipeline.args = SimpleNamespace(
        vpr_method="cosplace", vpr_backbone="ResNet18",
        vpr_descriptors_dimension=256, device="cpu",
    )
    with pytest.raises(ImportError):
        pipeline.init_vpr_model()


def _pinhole_K() -> np.ndarray:
    return np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])


def _project_points(K: np.ndarray, T_world_cam: np.ndarray, pts_world: np.ndarray):
    T_cam_world = np.linalg.inv(T_world_cam)
    pts_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
    pts_cam = (T_cam_world @ pts_h.T).T[:, :3]
    uv_h = (K @ pts_cam.T).T
    return uv_h[:, :2] / uv_h[:, 2:3], pts_cam[:, 2]


def _inside_frame(kpts: np.ndarray, w=640, h=480) -> np.ndarray:
    return (kpts[:, 0] >= 0) & (kpts[:, 0] < w) & (kpts[:, 1] >= 0) & (kpts[:, 1] < h)


def _make_node(trans, quat=(0.0, 0.0, 0.0, 1.0), img_size=(640, 480), node_id=0, depth_image=None):
    node = SimpleNamespace(
        id=node_id, trans=np.asarray(trans, dtype=np.float64), quat=np.asarray(quat, dtype=np.float64),
        raw_K=_pinhole_K(), raw_img_size=img_size, img_size=img_size,
        rgb_image=f"rgb_{node_id}", rgb_img_name=f"kf_{node_id}.jpg", depth_image=depth_image,
        mkpts=None, inliers=None,
    )

    def set_matched_kpts(mkpts, num_inliers):
        node.mkpts, node.inliers = mkpts, num_inliers

    def get_matched_kpts():
        return node.mkpts, node.inliers

    node.set_matched_kpts = set_matched_kpts
    node.get_matched_kpts = get_matched_kpts
    return node


def _make_bare_pipeline_no_ros(loc_pipeline_mod, img_matcher, gv_min_inliers=50):
    """没有 ROS 的 ``LocPipeline``(``__new__`` 绕开 ``__init__``,跟 run_vloc_offline_rerun.py
    的构造方式一样):真 PnP 求解器 + 替身 ``img_matcher``,足够跑通 ``localize_obs`` 的真实代码
    路径(不是把 ``_verify_one_candidate`` 也换成替身)。"""
    from utils.pose_solver import get_solver
    from rpe_default import cfg as _cfg

    pipeline = loc_pipeline_mod.LocPipeline.__new__(loc_pipeline_mod.LocPipeline)
    pipeline.args = SimpleNamespace(
        gv_min_inliers=gv_min_inliers, gv_min_inliers_nodepth=gv_min_inliers,
        save_img_matcher=False,
    )
    pipeline.img_matcher = img_matcher
    cfg_copy = _cfg.clone()
    cfg_copy.PNP.RANSAC_ITER = 1000
    cfg_copy.PNP.REPROJECTION_INLIER_THRESHOLD = 3
    cfg_copy.PNP.CONFIDENCE = 0.9999
    pipeline.pose_solver = get_solver("pnp", cfg_copy)
    return pipeline


def _synthetic_match(T_world_map, T_world_query, seed=0, n_pts=150):
    K = _pinhole_K()
    rng = np.random.default_rng(seed)
    pts_world = np.column_stack([
        rng.uniform(-1.0, 1.0, n_pts), rng.uniform(-0.8, 0.8, n_pts), rng.uniform(2.0, 5.0, n_pts),
    ])
    kpts_map, _ = _project_points(K, T_world_map, pts_world)
    kpts_query, depth_vals = _project_points(K, T_world_query, pts_world)
    keep = _inside_frame(kpts_map) & _inside_frame(kpts_query)
    kpts_map, kpts_query, depth_vals = kpts_map[keep], kpts_query[keep], depth_vals[keep]
    assert len(kpts_map) >= 60
    depth_img = np.zeros((480, 640), dtype=np.float64)
    px = np.clip(kpts_query[:, 0].astype(np.int32), 0, 639)
    py = np.clip(kpts_query[:, 1].astype(np.int32), 0, 479)
    depth_img[py, px] = depth_vals
    return kpts_map, kpts_query, depth_img


def test_localize_obs_global_mode_works_without_ros(loc_pipeline_mod_no_ros):
    """没有 ROS 时,LocPipeline.localize_obs(mode="global")用替身匹配器仍能正常核验并解出位姿。"""
    T_world_map = np.eye(4)
    T_world_query = np.eye(4)
    T_world_query[:3, 3] = [0.3, 0.0, 0.0]
    kpts_map, kpts_query, depth_img = _synthetic_match(T_world_map, T_world_query, seed=10)

    def fake_matcher(_map_rgb, _obs_rgb):
        return {"num_inliers": len(kpts_map), "inlier_kpts0": kpts_map, "inlier_kpts1": kpts_query}

    pipeline = _make_bare_pipeline_no_ros(loc_pipeline_mod_no_ros, fake_matcher, gv_min_inliers=50)
    pipeline.image_graph = SimpleNamespace(get_node=lambda nid: _make_node(trans=T_world_map[:3, 3], node_id=nid))
    depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)
    obs_node = _make_node(trans=T_world_query[:3, 3], node_id="obs", depth_image=depth_tensor)

    result = pipeline.localize_obs(obs_node, mode="global", prior={"candidate_ids": [7]})

    assert result.succ is True
    assert result.stage == "global"
    assert result.map_id == 7
    assert result.T_world_obs is not None
    t_err = np.linalg.norm(result.T_world_obs[:3, 3] - T_world_query[:3, 3])
    assert t_err < 0.05
    # import loc_pipeline 之后,除了咱们自己塞的 None 占位,没有谁偷偷 import 成功过 rospy。
    assert sys.modules.get("rospy") is None


def test_localize_obs_local_mode_works_without_ros(loc_pipeline_mod_no_ros):
    """没有 ROS 时,LocPipeline.localize_obs(mode="local", 给定 ref_map_node)也能正常跑完。"""
    T_world_map = np.eye(4)
    T_world_query = np.eye(4)
    T_world_query[:3, 3] = [0.2, 0.0, 0.0]
    kpts_map, kpts_query, depth_img = _synthetic_match(T_world_map, T_world_query, seed=11)

    def fake_matcher(_map_rgb, _obs_rgb):
        return {"num_inliers": len(kpts_map), "inlier_kpts0": kpts_map, "inlier_kpts1": kpts_query}

    pipeline = _make_bare_pipeline_no_ros(loc_pipeline_mod_no_ros, fake_matcher, gv_min_inliers=50)
    ref_node = _make_node(trans=T_world_map[:3, 3], node_id=3)
    depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)
    obs_node = _make_node(trans=T_world_query[:3, 3], node_id="obs", depth_image=depth_tensor)

    result = pipeline.localize_obs(obs_node, mode="local", prior={"ref_map_node": ref_node})

    assert result.succ is True
    assert result.stage == "local"
    assert result.map_id == 3
    assert result.T_world_obs is not None
    assert pipeline.ref_map_node is ref_node
    # perform_local_loc 原来的行为(不管核验过没过都记匹配点,供可视化用)依然保留。
    assert ref_node.get_matched_kpts()[1] == len(kpts_map)
    assert obs_node.get_matched_kpts()[1] == len(kpts_map)
