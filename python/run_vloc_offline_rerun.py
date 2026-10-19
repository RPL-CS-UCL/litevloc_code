#!/usr/bin/env python3
"""Run LiteVLoc offline localization without ROS, writing results to a Rerun .rrd file.

Usage
-----
cd /Titan/code/robohike_ws/src/opennavmap
LD_LIBRARY_PATH=/root/miniconda3/envs/opennavmap/lib:$LD_LIBRARY_PATH \
PYTHONPATH=third_party/litevloc_code/python:third_party/VPR-methods-evaluation \
python third_party/litevloc_code/python/run_vloc_offline_rerun.py \
  --map_path /Titan/dataset/data_opennavmap/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
  --query_data_path /Titan/dataset/data_opennavmap/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
  --output_rrd third_party/litevloc_code/logs/vloc_s17DRP5sb8fy.rrd \
  --image_size 512 288 --device cuda \
  --vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 256 \
  --vpr_match_model single_match \
  --img_matcher master --pose_solver pnp \
  --config_pose_solver third_party/litevloc_code/python/config/dataset/matterport3d.yaml \
  --global_pos_threshold 10.0

--image_size / --n_kpts / --min_master_conf_thre / --gv_min_inliers(_nodepth) 不给就按
--img_matcher 从 config/matcher_profiles.yaml 读(M5-B4);命令行显式给的仍然优先。

Modes (--mode)
--------------
seq (默认，一字不改):跟原来一样，连续 5 帧的描述子做序列匹配(--vpr_match_model)拿全局定位候选，
    候选核验(统一核验:网格去重 + PnP-RANSAC 内点数,见 utils/geom_verify.py)遇到第一个过门槛就停，
    核验时顺带解出的位姿直接当这一帧的结果(M5-B4,不再额外跑一次局部匹配);查询没深度时核验只认出
    关键帧、退化走"最近关键帧 + 邻居里挑描述子最近的 → 匹配 → 统一核验"那一段精化位姿。
reloc:不再攒 5 帧，每一帧都单独做一次全局定位(VPR 换成单帧最近邻 single_match)，其余(候选核验、
    局部匹配)跟 seq 一样。
track:模仿在线节点 ros_loc_pipeline/loc_pipeline 的切换方式——第一帧、或者上一帧没能定位成功时，
    跟 reloc 一样做一次全局定位；上一帧定位成功了就跳过这次的全局定位(VPR)，直接把上一帧估计出来
    的位姿当先验，走局部匹配那一段。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_LITEVLOC_ROOT = _HERE.parent
_VPR_EVAL = _LITEVLOC_ROOT.parent / "VPR-methods-evaluation"
sys.path.insert(0, str(_VPR_EVAL))
sys.path.insert(0, str(_HERE))

# 下面这段 import 顺序不是随便排的：Jetson 上 native .so 一多就会把 glibc 的
# static TLS block 占满（sklearn 自带的 libgomp 是重灾区，报错是
# "cannot allocate memory in static TLS block"），utils_vpr_method 必须排在
# utils_rerun 前面才不会撞上这个坑（之前 ruff --fix 重排过一次，复现了这个问题，
# 这里手动按原顺序排回来）。改这段前先跑一下 `python run_vloc_offline_rerun.py --help`。
from image_graph import ImageGraphLoader  # noqa: I001
from image_node import ImageNode
from utils.utils_geom import (
    read_intrinsics, read_poses, read_descriptors,
    convert_pose_inv, convert_vec_to_matrix, convert_matrix_to_vec,
    correct_intrinsic_scale, compute_pose_error,
)
from utils.utils_image import load_rgb_image, load_depth_image, to_numpy
from utils.utils_vpr_method import initialize_vpr_model, initialize_match_model
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_pipeline import GV_SCORE_THRESHOLD, load_matcher_profile
from utils.pose_solver import get_solver
from utils.utils_rerun import (
    init_rerun, save_rrd,
    log_world_frame_axes,
    log_map_nodes, log_map_edges,
    set_frame_time, log_query_image, log_query_camera,
    log_trajectory, log_image_matching,
)
from rpe_default import cfg
# M5-B4b:候选核验(候选按排名逐个核验、早停、取最高分)和局部定位的关键帧挑选不再在这个脚本里
# 单独抄一份，转发到 LocPipeline.localize_obs(跟在线节点 ros_loc_pipeline.py 共用同一个实现，
# 见 utils/geom_verify.py 的 verify_match_result 文档字符串)。
from loc_pipeline import LocPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiteVLoc offline localization to Rerun .rrd")
    p.add_argument("--map_path", required=True)
    p.add_argument("--query_data_path", required=True)
    p.add_argument(
        "--output_rrd", type=pathlib.Path,
        default=_HERE.parent / "logs" / "vloc_result.rrd",
    )
    p.add_argument("--image_size", nargs=2, type=int, default=None,
                   help="不给就按 --img_matcher 从 config/matcher_profiles.yaml 读(兜底 512x288)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vpr_method", default="cosplace")
    p.add_argument("--vpr_backbone", default="ResNet18")
    p.add_argument("--vpr_descriptors_dimension", type=int, default=256)
    p.add_argument("--vpr_match_model", default="vpr_dp")
    p.add_argument("--vpr_match_seq_len", type=int, default=5)
    p.add_argument("--img_matcher", default="master")
    p.add_argument("--n_kpts", type=int, default=None,
                   help="不给就按 --img_matcher 从 config/matcher_profiles.yaml 读(兜底 2048)")
    p.add_argument("--pose_solver", default="pnp")
    p.add_argument("--config_pose_solver", default="matterport3d.yaml")
    p.add_argument("--global_pos_threshold", type=float, default=10.0)
    p.add_argument("--min_master_conf_thre", type=float, default=None,
                   help="不给就按 --img_matcher 从 config/matcher_profiles.yaml 的 matcher_kwargs.min_conf_thr 读")
    p.add_argument("--gv_min_inliers", type=float, default=None,
                   help="M5-B4 统一核验门槛(查询图网格去重后,查询深度 PnP-RANSAC 的内点数);"
                        "不给就按 --img_matcher 从 config/matcher_profiles.yaml 读(兜底 100)")
    p.add_argument("--gv_min_inliers_nodepth", type=float, default=None,
                   help="同上,但用于查询没有深度时的基础矩阵核验分数(utils.geom_verify.verify_fundamental)")
    p.add_argument("--matcher_config", type=str, default=None,
                   help="matcher_profiles.yaml 的路径,不给就用 config/matcher_profiles.yaml(随本仓库)")
    p.add_argument("--min_solver_inliers_thre", type=int, default=None,
                   help="[M5-B4 起未使用] 'localized' 判定已经改用 --gv_min_inliers;这个参数只留着接住"
                        "旧命令行(比如 scripts/run_vloc_offline_rerun.sh)还传的值,不会报 unrecognized "
                        "argument,但传了也不生效")
    p.add_argument("--depth_scale", type=float, default=0.001)
    p.add_argument(
        "--mode", choices=["seq", "reloc", "track"], default="seq",
        help="seq(默认,行为不变)=5帧序列匹配; reloc=每帧单独全局定位; "
             "track=首帧/丢失后做 reloc,其余帧跳过VPR、用上一帧估计位姿当先验走局部匹配+PnP",
    )
    p.add_argument(
        "--gv_early_exit", type=int, choices=[0, 1], default=1,
        help="候选核验遇到第一个过门槛就停(1,M5-B4 起在线/离线都是这个默认行为) / 全部候选跑完"
             "取核验分数最高的那个(0,在线节点改早停之前的行为,用来对照)",
    )
    p.add_argument(
        "--results_json", type=pathlib.Path, default=None,
        help="optional: write per-query records + a summary as JSON (errors are kept even when "
             "verify_inliers is below --gv_min_inliers, so other thresholds can be studied)",
    )
    args = p.parse_args()

    # M5-B4:按 --img_matcher 从 config/matcher_profiles.yaml 填命令行没显式给的参数(显式给的优先)。
    profile = load_matcher_profile(args.img_matcher, args.matcher_config)
    if args.image_size is None:
        img_size = profile.get("image_size")
        args.image_size = list(img_size) if img_size is not None else [512, 288]
    if args.n_kpts is None:
        args.n_kpts = profile.get("n_kpts", 2048)
    if args.gv_min_inliers is None:
        args.gv_min_inliers = profile.get("gv_min_inliers", GV_SCORE_THRESHOLD)
    if args.gv_min_inliers_nodepth is None:
        args.gv_min_inliers_nodepth = profile.get("gv_min_inliers_nodepth", GV_SCORE_THRESHOLD)
    matcher_kwargs = dict(profile.get("matcher_kwargs", {}) or {})
    if args.min_master_conf_thre is not None:
        matcher_kwargs["min_conf_thr"] = args.min_master_conf_thre
    elif "min_conf_thr" in matcher_kwargs:
        args.min_master_conf_thre = matcher_kwargs["min_conf_thr"]
    args.matcher_kwargs = matcher_kwargs

    return args


class _ProfilingImgMatcher:
    """包一层在 ``img_matcher`` 外面，只用来接住匹配器每次调用自己写的 ``last_profile``
    (逐段计时，比如 EDM/MASt3R 的编码耗时)。

    M5-B4b：候选核验(含早停/取最高分)和局部定位的关键帧挑选都挪进了 ``LocPipeline.localize_obs``
    (跟在线节点共用实现)，这个脚本不再自己跑那个候选循环，也就摸不到循环内部每次匹配的
    ``last_profile`` 了——``LocResult`` 没有带上这个字段(在线节点不需要它，不该为了离线重跑的
    诊断信息改在线节点的返回值)。这层包装照旧把 ``img_matcher`` 转发给 ``localize_obs`` 内部
    真正调用匹配器的地方，顺手把每次调用后的 ``last_profile`` 收集起来，供 ``results_json`` 的
    每帧 ``profile`` 字段用；除此之外行为和原始 ``img_matcher`` 完全一样。
    """

    def __init__(self, matcher):
        self._matcher = matcher
        self.collected: list = []

    def __call__(self, *args, **kwargs):
        result = self._matcher(*args, **kwargs)
        last_profile = getattr(self._matcher, "last_profile", None)
        if last_profile:
            self.collected.append(dict(last_profile))
        return result

    def drain_profiles(self) -> list:
        """取走并清空目前收集到的 profile(每帧用一次,跟旧版 verify_candidates 每帧重新
        计一份 profiles 列表的语义一致)。"""
        profiles, self.collected = self.collected, []
        return profiles

    def __getattr__(self, name):
        # 除了 __call__/last_profile(上面已经处理),其余属性原样转发给真正的匹配器。
        return getattr(self._matcher, name)


def estimate_local_pose(
    pose_solver,
    ref_map_node,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    raw_K: np.ndarray,
    width: int,
    height: int,
    resize: tuple,
    depth_img,
) -> dict | None:
    """ref_map_node<->query 观测 的匹配点换回原图尺寸后解 PnP。

    离线重跑(seq/reloc/track 三种模式共用)和 tools/vloc_eval/matcher_pair_bench.py 共用这一段，
    别抄第二份。``ref_map_node`` 只用到 ``raw_img_size`` / ``img_size`` / ``raw_K`` / ``trans`` /
    ``quat`` 五个字段，调用方不一定要传真正的 ``ImageNode``，随便一个有这五个字段的对象
    (比如 ``types.SimpleNamespace``)都行。返回 None 表示抛了异常（深度缺失、PnP 数值问题等，
    调用方按原来的行为打日志即可跳过）；否则返回 {"solver_inliers", "trans", "quat"}，
    trans/quat 在没解出有效位姿时是 None。
    """
    try:
        w_r = ref_map_node.raw_img_size[0] / ref_map_node.img_size[0]
        h_r = ref_map_node.raw_img_size[1] / ref_map_node.img_size[1]
        mkpts0_raw = mkpts0 * [w_r, h_r]
        w_q = width / resize[0]
        h_q = height / resize[1]
        mkpts1_raw = mkpts1 * [w_q, h_q]
        depth_np = to_numpy(depth_img.squeeze(0))
        R_est, t_est, n_sol = pose_solver.estimate_pose(
            mkpts1_raw, mkpts0_raw,
            raw_K, ref_map_node.raw_K,
            depth_np, None,
        )
        result = {"solver_inliers": int(n_sol), "trans": None, "quat": None}
        if n_sol > 0 and np.all(np.isfinite(R_est)):
            T_wm = convert_vec_to_matrix(ref_map_node.trans, ref_map_node.quat, "xyzw")
            T_mo = np.eye(4)
            T_mo[:3, :3], T_mo[:3, 3] = R_est, t_est.reshape(3)
            T_wo = T_wm @ T_mo
            trans_est, quat_est = convert_matrix_to_vec(T_wo, "xyzw")
            result["trans"], result["quat"] = trans_est, quat_est
        return result
    except Exception as exc:
        print(f"Pose solve: {exc}")
        return None


def summarize(records: list) -> dict:
    """Localized ratio, error percentiles, and per-frame/per-stage timing percentiles."""
    loc = [r for r in records if r["localized"]]
    t_err = np.array([r["t_err"] for r in loc], dtype=float)
    r_err = np.array([r["r_err"] for r in loc], dtype=float)
    time_s = np.array([r["time_s"] for r in records], dtype=float)

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else float("nan")

    by_stage: dict = {}
    for r in records:
        by_stage.setdefault(r.get("stage", "none"), []).append(r["time_s"])
    time_by_stage = {
        stage: {"median_s": pct(np.array(ts), 50), "p90_s": pct(np.array(ts), 90), "n": len(ts)}
        for stage, ts in by_stage.items()
    }

    gpu_mems = [r["gpu_mem_peak_mb"] for r in records if r.get("gpu_mem_peak_mb") is not None]

    return {
        "num_queries": len(records),
        "num_vpr_verified": sum(r["vpr_verified"] for r in records),
        "num_localized": len(loc),
        "localized_ratio": len(loc) / len(records) if records else float("nan"),
        "t_err_median_m": pct(t_err, 50), "t_err_p90_m": pct(t_err, 90),
        "r_err_median_deg": pct(r_err, 50), "r_err_p90_deg": pct(r_err, 90),
        "time_per_query_mean_s": float(np.mean(time_s)) if records else float("nan"),
        "time_s_median": pct(time_s, 50),
        "time_s_p90": pct(time_s, 90),
        "time_by_stage": time_by_stage,
        "gpu_mem_peak_mb_max": float(max(gpu_mems)) if gpu_mems else None,
    }


def get_loadable_query_keys(
    query_root: pathlib.Path,
    poses: dict,
    intrs: dict,
    descs: dict,
) -> list:
    keys = []
    for key in poses:
        if not (query_root / key).exists():
            continue
        if key not in intrs:
            continue
        if descs is not None and key not in descs:
            continue
        keys.append(key)
    return keys


def main() -> None:
    args = parse_args()

    init_rerun("litevloc_offline_vloc")

    map_root = pathlib.Path(args.map_path)
    resize = tuple(args.image_size)
    config = dict(
        resize=resize, depth_scale=args.depth_scale,
        load_rgb=True, load_depth=False, normalized=False,
    )

    image_graph = ImageGraphLoader.load_data(
        map_root=map_root,
        resize=config["resize"],
        depth_scale=config["depth_scale"],
        load_rgb=config["load_rgb"],
        load_depth=config["load_depth"],
        normalized=config["normalized"],
        edge_type="covis",
    )
    print(str(image_graph))

    log_world_frame_axes()
    log_map_nodes(image_graph)
    log_map_edges(image_graph, edge_type="covis")

    trav_graph = ImageGraphLoader.load_data(
        map_root=map_root,
        resize=config["resize"],
        depth_scale=config["depth_scale"],
        load_rgb=False, load_depth=False, normalized=False,
        edge_type="trav",
    )
    log_map_edges(trav_graph, edge_type="trav")

    vpr_model = initialize_vpr_model(
        args.vpr_method, args.vpr_backbone,
        args.vpr_descriptors_dimension, args.device,
    )
    db_node_ids = [node.id for node in image_graph.nodes.values()]
    db_descriptors = np.array(
        [node.get_descriptor() for node in image_graph.nodes.values()],
        dtype="float32",
    )
    db_poses = np.empty((image_graph.get_num_node(), 7), dtype="float32")
    for i, (_, node) in enumerate(image_graph.nodes.items()):
        db_poses[i, :3] = node.trans
        db_poses[i, 3:] = node.quat

    # reloc/track 两种模式每帧单独做全局定位，VPR 换成单帧最近邻（不需要攒 5 帧）；
    # seq 模式(默认)行为不变，用 --vpr_match_model 指定的序列匹配。
    if args.mode == "seq":
        vpr_match_model = initialize_match_model(args.vpr_match_model, args.vpr_match_seq_len)
    else:
        vpr_match_model = initialize_match_model("single_match", 1)
    vpr_match_model.initialize_model(db_descriptors)

    # M5-B4:matcher_kwargs(比如 master 系列的 min_conf_thr、edm 的 ransac_model)在构造时就传
    # 给 get_matcher,不再构造完事后按 hasattr(img_matcher, "min_conf_thr") 判断、事后赋值。
    img_matcher = initialize_img_matcher(
        args.img_matcher, args.device, args.n_kpts, matcher_kwargs=args.matcher_kwargs
    )

    cfg.merge_from_file(args.config_pose_solver)
    pose_solver = get_solver(args.pose_solver, cfg)

    # M5-B4b:候选核验(早停/取最高分)和局部定位的关键帧挑选都转发到 LocPipeline.localize_obs,
    # 跟在线节点(ros_loc_pipeline.py)共用同一份实现,这个脚本不再自己抄一份。用 __new__ 绕开
    # LocPipeline.__init__(那是给在线节点用的:建 out_dir/log_dir、ROS 发布器,离线重跑不需要),
    # 只手动装上 localize_obs / search_keyframe_from_graph 真正用到的几个属性——这是离线用法的
    # "干净构造方式",不加新的命令行开关。
    args.save_img_matcher = False  # loc_pipeline.perform_image_matching 的可视化开关,离线重跑不用
    loc = LocPipeline.__new__(LocPipeline)
    loc.args = args
    loc.image_graph = image_graph
    loc.DB_Node_IDS = db_node_ids
    loc.DB_POSES = db_poses
    loc.pose_solver = pose_solver
    loc.img_matcher = _ProfilingImgMatcher(img_matcher)
    loc.ref_map_node = None

    query_root = pathlib.Path(args.query_data_path)
    poses = read_poses(str(query_root / "poses.txt"))
    intrs = read_intrinsics(str(query_root / "intrinsics.txt"))
    descs = read_descriptors(str(query_root / "database_descriptors.txt"))
    query_keys = get_loadable_query_keys(query_root, poses, intrs, descs)
    print(f"Loadable query frames: {len(query_keys)}")

    use_cuda_mem = args.device.startswith("cuda") and torch.cuda.is_available()

    traj_est, traj_gt = [], []
    records = []
    curr_query_descs = []
    has_global_pos = False
    ref_map_node = None
    # track 模式用：上一帧定位成功时的估计位姿，下一帧跳过 VPR 直接拿来当先验
    prev_trans, prev_quat = None, None

    for frame_id, rgb_img_name in enumerate(query_keys):
        print(f"Loading observation seq/{rgb_img_name}")
        pose = poses[rgb_img_name]
        intr = intrs[rgb_img_name]
        width, height = int(intr[4]), int(intr[5])
        raw_K = np.array(
            [intr[0], 0, intr[2], 0, intr[1], intr[3], 0, 0, 1],
            dtype=np.float32,
        ).reshape(3, 3)
        K = correct_intrinsic_scale(raw_K, resize[0] / width, resize[1] / height)
        img_size = np.array([resize[0], resize[1]])

        rgb_img = load_rgb_image(str(query_root / rgb_img_name), resize)
        depth_img_path = str(query_root / rgb_img_name.replace("color.jpg", "depth.png"))
        depth_img = (
            load_depth_image(depth_img_path, depth_scale=args.depth_scale)
            if os.path.exists(depth_img_path)
            else None
        )

        trans_gt, quat_gt = convert_pose_inv(pose[4:], np.roll(pose[:4], -1), "xyzw")

        obs_node = ImageNode(
            frame_id, rgb_img, depth_img, descs[rgb_img_name].reshape(1, -1),
            float(frame_id),
            np.zeros(3), np.array([0, 0, 0, 1]),
            K, img_size,
            rgb_img_name, rgb_img_name.replace("color.jpg", "depth.png"),
        )
        obs_node.set_raw_intrinsics(raw_K, (width, height))
        obs_node.set_pose_gt(trans_gt, quat_gt)

        set_frame_time(frame_id, float(frame_id))
        rgb_np = (np.transpose(to_numpy(rgb_img), (1, 2, 0)) * 255).astype(np.uint8)
        log_query_image(rgb_np)
        log_query_camera(trans_gt, quat_gt, K, img_size, is_gt=True)

        if use_cuda_mem:
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        rec = {
            "key": rgb_img_name, "frame_id": frame_id, "gt_trans": trans_gt.tolist(),
            "mode": args.mode, "stage": "none",
            "vpr_verified": False, "vpr_max_inliers": None, "ref_node": None,
            "match_inliers": None, "solver_inliers": None, "verify_inliers": None, "localized": False,
            "est_trans": None, "t_err": None, "r_err": None,
            "t_vpr_s": 0.0, "t_gv_s": 0.0, "n_gv_matches": 0, "t_local_s": 0.0,
            "gpu_mem_peak_mb": None, "profile": [],
        }
        query_desc = descs[rgb_img_name].reshape(1, -1)

        # M5-B4b:统一核验门槛(查询有没有深度决定用哪把尺子:PnP / 基础矩阵)现在由
        # LocPipeline.localize_obs 自己从 obs_node.depth_image 推(跟这里 depth_img 是否为 None
        # 完全一致),这个脚本不用再自己算一遍 has_depth/gv_threshold。

        # track 模式：上一帧定位成功就跳过这帧的全局定位(VPR)，直接用上一帧估计位姿当先验。
        track_continue = args.mode == "track" and has_global_pos and prev_trans is not None
        do_global = not track_continue

        if track_continue:
            obs_node.set_pose(prev_trans, prev_quat)
        elif args.mode == "seq":
            curr_query_descs.append(query_desc)
            curr_query_descs = curr_query_descs[-args.vpr_match_seq_len:]
            if len(curr_query_descs) < args.vpr_match_seq_len:
                do_global = False
                has_global_pos = False
                print("Not enough query descs for seq match")

        if do_global:
            rec["stage"] = "global"
            t_vpr0 = time.time()
            if args.mode == "seq":
                query_descs_stack = np.array(curr_query_descs).reshape(-1, query_desc.shape[1])
                recall_preds, _, _ = vpr_match_model.match(query_descs_stack, recall_values=5)
            else:
                recall_preds, _, _ = vpr_match_model.match(query_desc, recall_values=5)
            rec["t_vpr_s"] = time.time() - t_vpr0
            print(recall_preds)

            # M5-B4b:候选核验(早停/取最高分)转发到 LocPipeline.localize_obs(mode="global"),
            # 跟在线节点共用同一份实现,这个脚本不再自己跑这段候选循环。
            candidate_ids = [db_node_ids[pred] for pred in recall_preds]
            result = loc.localize_obs(
                obs_node, mode="global",
                prior={"candidate_ids": candidate_ids, "early_exit": bool(args.gv_early_exit)},
            )
            rec["vpr_max_inliers"] = int(result.verify_inliers)
            rec["verify_inliers"] = int(result.verify_inliers)
            rec["t_gv_s"] = result.t_verify_s
            rec["n_gv_matches"] = result.n_matches_run
            rec["profile"].extend(loc.img_matcher.drain_profiles())

            if result.succ:
                rec["vpr_verified"] = True
                has_global_pos = True
                ref_map_node = image_graph.get_node(result.map_id)
                if result.T_world_obs is not None:
                    # M5-B4:核验时已经用 PnP 解出了位姿,直接当这一帧的结果用,不用再跑一次
                    # 局部匹配(跟在线节点 ros_loc_pipeline.process_frame 的新行为一致)。
                    trans_est, quat_est = convert_matrix_to_vec(result.T_world_obs, "xyzw")
                    t_err, r_err = compute_pose_error((trans_est, quat_est), (trans_gt, quat_gt), mode="vector")
                    rec.update(
                        est_trans=trans_est.tolist(), t_err=float(t_err), r_err=float(r_err),
                        solver_inliers=int(result.verify_inliers), ref_node=int(result.map_id),
                        localized=True,
                    )
                    obs_node.set_pose(trans_est, quat_est)
                    print(f"[Succ] global verify solved pose directly, verify_inliers={result.verify_inliers}")
                    print(f"Groundtruth Poses: {trans_gt}")
                    print(f"Estimated Poses: {trans_est}")
                    log_query_camera(trans_est, quat_est, obs_node.K, obs_node.img_size, rgb_image=rgb_np, is_gt=False)
                    traj_est.append(trans_est.copy())
                    print(f"t_err={t_err:.3f}m r_err={r_err:.2f}deg verify_inliers={result.verify_inliers}")
                else:
                    # 查询没有深度:核验只认出了关键帧,没法解位姿;位姿先用关键帧的位姿当粗先验,
                    # 交给下面的局部定位精化。
                    obs_node.set_pose(ref_map_node.trans, ref_map_node.quat)
                    print(f"Found VPR Node in global position: {result.map_id}")
            else:
                has_global_pos = False
                ref_map_node = None
                print(f"VPR fail, inliers={result.verify_inliers}")

        # M5-B4:全局核验已经直接解出位姿(rec["localized"] 已是 True)时,跳过局部定位——
        # 跟在线节点一样,"第一个统一核验分数达标的候选就停,不再做第二次局部匹配"。
        if has_global_pos and ref_map_node is not None and not rec["localized"]:
            rec["stage"] = "local"
            # M5-B4b:局部定位的关键帧挑选(最近关键帧 + 邻居里挑描述子最近的)和核验/解位姿都转发
            # 到 LocPipeline.localize_obs(mode="local")——跟在线节点共用同一份 search_keyframe_
            # from_graph 实现(这份比这个脚本原来自己写的那份多一个 90° 角度过滤,见 M5-B4b 等价性
            # 核对),不再是这个脚本自己抄一份"按位姿 KNN 找最近关键帧"。不传 prior["ref_map_node"],
            # 让它自己挑(跟 ros_loc_pipeline.process_frame 没给先验时的行为一致)。
            t_local0 = time.time()
            local_result = loc.localize_obs(obs_node, mode="local")
            rec["t_local_s"] = time.time() - t_local0
            rec["profile"].extend(loc.img_matcher.drain_profiles())

            # n_matches_run == 0 表示 search_keyframe_from_graph 一个候选都没挑到(极少见:连最近
            # 关键帧本身都不满足距离/角度门槛),这时没有匹配结果可记录,ref_map_node 保持上一帧的值。
            if local_result.n_matches_run > 0:
                ref_map_node = loc.ref_map_node
                mkpts0, _ = ref_map_node.get_matched_kpts()
                mkpts1, num_inliers = obs_node.get_matched_kpts()
                rec["ref_node"], rec["match_inliers"] = int(ref_map_node.id), int(num_inliers)
                print(f"Number of matched inliers: {num_inliers}")

                ref_rgb = (np.transpose(to_numpy(ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
                log_image_matching(ref_rgb, rgb_np, mkpts0, mkpts1, node_id=ref_map_node.id)

            rec["solver_inliers"] = local_result.verify_inliers
            rec["verify_inliers"] = local_result.verify_inliers
            if local_result.T_world_obs is not None:
                trans_est, quat_est = convert_matrix_to_vec(local_result.T_world_obs, "xyzw")
                # the error is recorded even below gv_threshold (study of other thresholds)
                t_err, r_err = compute_pose_error(
                    (trans_est, quat_est), (trans_gt, quat_gt), mode="vector"
                )
                rec.update(est_trans=trans_est.tolist(), t_err=float(t_err), r_err=float(r_err))
                if local_result.succ:
                    rec["localized"] = True
                    obs_node.set_pose(trans_est, quat_est)
                    print(f"[Succ] sufficient number {local_result.verify_inliers} verify_inliers")
                    print(f"Groundtruth Poses: {trans_gt}")
                    print(f"Estimated Poses: {trans_est}")
                    log_query_camera(trans_est, quat_est, obs_node.K, obs_node.img_size, rgb_image=rgb_np, is_gt=False)
                    traj_est.append(trans_est.copy())
                    print(f"t_err={t_err:.3f}m r_err={r_err:.2f}deg verify_inliers={local_result.verify_inliers}")

        if args.mode == "track":
            if rec["localized"]:
                prev_trans, prev_quat = obs_node.trans, obs_node.quat
                has_global_pos = True
            else:
                has_global_pos = False
                ref_map_node = None

        rec["time_s"] = time.time() - t_start
        if use_cuda_mem:
            rec["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        records.append(rec)
        traj_gt.append(trans_gt.copy())
        log_trajectory(traj_gt, is_gt=True)
        if traj_est:
            log_trajectory(traj_est, is_gt=False)

    args.output_rrd.parent.mkdir(parents=True, exist_ok=True)
    save_rrd(args.output_rrd)
    print(f"Saved to {args.output_rrd}")
    if args.results_json is not None:
        summary = summarize(records)
        print(f"Summary: {summary}")
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"args": {k: str(v) for k, v in vars(args).items()}, "summary": summary, "records": records}
        args.results_json.write_text(json.dumps(payload, indent=1))
        print(f"Saved results to {args.results_json}")


if __name__ == "__main__":
    main()
