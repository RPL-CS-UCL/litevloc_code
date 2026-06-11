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
  --output_rrd third_party/litevloc_code/output/vloc_s17DRP5sb8fy.rrd \
  --image_size 512 288 --device cuda \
  --vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 256 \
  --vpr_match_model single_match \
  --img_matcher master --pose_solver pnp \
  --config_pose_solver third_party/litevloc_code/python/config/dataset/matterport3d.yaml \
  --global_pos_threshold 10.0 --min_master_conf_thre 1.5 --min_solver_inliers_thre 200
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_LITEVLOC_ROOT = _HERE.parent
_VPR_EVAL = _LITEVLOC_ROOT.parent / "VPR-methods-evaluation"
sys.path.insert(0, str(_VPR_EVAL))
sys.path.insert(0, str(_HERE))

from image_graph import ImageGraphLoader
from image_node import ImageNode
from utils.utils_geom import (
    read_intrinsics, read_poses, read_descriptors,
    convert_pose_inv, convert_vec_to_matrix, convert_matrix_to_vec,
    correct_intrinsic_scale, compute_pose_error,
)
from utils.utils_image import load_rgb_image, load_depth_image, to_numpy
from utils.utils_vpr_method import initialize_vpr_model, initialize_match_model, perform_knn_search, compute_euclidean_dis
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_pipeline import GV_SCORE_THRESHOLD
from utils.pose_solver import get_solver
from utils.utils_rerun import (
    init_rerun, save_rrd,
    log_world_frame_axes,
    log_map_nodes, log_map_edges,
    set_frame_time, log_query_image, log_query_camera,
    log_trajectory, log_image_matching, log_pose_axes,
)
from benchmark_rpe.rpe_default import cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiteVLoc offline localization to Rerun .rrd")
    p.add_argument("--map_path", required=True)
    p.add_argument("--query_data_path", required=True)
    p.add_argument(
        "--output_rrd", type=pathlib.Path,
        default=_HERE.parent / "output" / "vloc_result.rrd",
    )
    p.add_argument("--image_size", nargs=2, type=int, default=[512, 288])
    p.add_argument("--device", default="cuda")
    p.add_argument("--vpr_method", default="cosplace")
    p.add_argument("--vpr_backbone", default="ResNet18")
    p.add_argument("--vpr_descriptors_dimension", type=int, default=256)
    p.add_argument("--vpr_match_model", default="sequence_match")
    p.add_argument("--vpr_match_seq_len", type=int, default=5)
    p.add_argument("--img_matcher", default="master")
    p.add_argument("--n_kpts", type=int, default=2048)
    p.add_argument("--pose_solver", default="pnp")
    p.add_argument("--config_pose_solver", default="matterport3d.yaml")
    p.add_argument("--global_pos_threshold", type=float, default=10.0)
    p.add_argument("--min_master_conf_thre", type=float, default=1.5)
    p.add_argument("--min_solver_inliers_thre", type=int, default=200)
    p.add_argument("--depth_scale", type=float, default=0.001)
    return p.parse_args()


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
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
    logging.info(str(image_graph))
    
    log_world_frame_axes(length=0.5, radii=0.02)
    log_map_nodes(image_graph)

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

    vpr_match_model = initialize_match_model(args.vpr_match_model, args.vpr_match_seq_len)
    vpr_match_model.initialize_model(db_descriptors)

    img_matcher = initialize_img_matcher(args.img_matcher, args.device, args.n_kpts)
    if args.img_matcher == "master":
        img_matcher.min_conf_thr = args.min_master_conf_thre

    cfg.merge_from_file(args.config_pose_solver)
    pose_solver = get_solver(args.pose_solver, cfg)

    query_root = pathlib.Path(args.query_data_path)
    poses = read_poses(str(query_root / "poses.txt"))
    intrs = read_intrinsics(str(query_root / "intrinsics.txt"))
    descs = read_descriptors(str(query_root / "database_descriptors.txt"))
    query_keys = get_loadable_query_keys(query_root, poses, intrs, descs)
    logging.info(f"Loadable query frames: {len(query_keys)}")

    traj_est, traj_gt = [], []
    curr_query_descs = []
    has_global_pos = False
    ref_map_node = None

    for frame_id, rgb_img_name in enumerate(query_keys):
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
        log_query_camera(trans_gt, quat_gt, raw_K, (width, height), is_gt=True)

        t_start = time.time()
        query_desc = descs[rgb_img_name].reshape(1, -1)
        curr_query_descs.append(query_desc)
        curr_query_descs = curr_query_descs[-args.vpr_match_seq_len:]

        if len(curr_query_descs) >= args.vpr_match_seq_len:
            query_descs_stack = np.array(curr_query_descs).reshape(-1, query_desc.shape[1])
            recall_preds, _, _ = vpr_match_model.match(query_descs_stack, recall_values=5)

            best_map_id, max_inliers = db_node_ids[recall_preds[0]], 0
            for pred in recall_preds:
                map_node = image_graph.get_node(db_node_ids[pred])
                result = img_matcher(map_node.rgb_image, obs_node.rgb_image)
                n = result["num_inliers"]
                if n > max_inliers:
                    best_map_id, max_inliers = db_node_ids[pred], n
                if n >= GV_SCORE_THRESHOLD:
                    ref_map_node = image_graph.get_node(db_node_ids[pred])
                    break

            if max_inliers >= GV_SCORE_THRESHOLD:
                has_global_pos = True
                ref_map_node = image_graph.get_node(best_map_id)
                obs_node.set_pose(ref_map_node.trans, ref_map_node.quat)
                logging.info(f"[frame {frame_id}] VPR → node {best_map_id} ({t_start:.1f}s)")
            else:
                has_global_pos = False
                ref_map_node = None
                logging.warning(f"[frame {frame_id}] VPR fail, inliers={max_inliers}")
        else:
            has_global_pos = False
            logging.warning(f"[frame {frame_id}] Not enough query descs for seq match")

        if has_global_pos and ref_map_node is not None:
            dis, pred = perform_knn_search(db_poses[:, :3], obs_node.trans.reshape(1, 3), 3, [1])
            if len(pred[0]) > 0:
                closest = image_graph.get_node(db_node_ids[pred[0][0]])
                candidates = [
                    n for n, _ in closest.edges.values()
                    if n.compute_distance(obs_node)[0] < args.global_pos_threshold
                ] + [closest]

                if candidates:
                    dists = [compute_euclidean_dis(obs_node.get_descriptor(), n.get_descriptor()) for n in candidates]
                    ref_map_node = candidates[np.argmin(dists)]

            match_result = img_matcher(ref_map_node.rgb_image, obs_node.rgb_image)
            mkpts0 = match_result["inlier_kpts0"]
            mkpts1 = match_result["inlier_kpts1"]
            num_inliers = match_result["num_inliers"]

            ref_map_node.set_matched_kpts(mkpts0, num_inliers)
            obs_node.set_matched_kpts(mkpts1, num_inliers)

            ref_rgb = (np.transpose(to_numpy(ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
            log_image_matching(ref_rgb, rgb_np, mkpts0, mkpts1, node_id=ref_map_node.id)

            if depth_img is not None:
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
                    if n_sol >= args.min_solver_inliers_thre:
                        T_wm = convert_vec_to_matrix(ref_map_node.trans, ref_map_node.quat, "xyzw")
                        T_mo = np.eye(4)
                        T_mo[:3, :3], T_mo[:3, 3] = R_est, t_est.reshape(3)
                        T_wo = T_wm @ T_mo
                        trans_est, quat_est = convert_matrix_to_vec(T_wo, "xyzw")
                        obs_node.set_pose(trans_est, quat_est)
                        log_query_camera(trans_est, quat_est, raw_K, (width, height), is_gt=False)
                        log_pose_axes(trans_est, quat_est, entity_path="query/pose_estimated/axes")
                        traj_est.append(trans_est.copy())
                        t_err, r_err = compute_pose_error(
                            (trans_est, quat_est), (trans_gt, quat_gt), mode="vector"
                        )
                        logging.info(f"[frame {frame_id}] t_err={t_err:.3f}m r_err={r_err:.2f}° inliers={n_sol}")
                except Exception as exc:
                    logging.warning(f"[frame {frame_id}] Pose solve: {exc}")

        traj_gt.append(trans_gt.copy())
        log_trajectory(traj_gt, is_gt=True)
        if traj_est:
            log_trajectory(traj_est, is_gt=False)

    args.output_rrd.parent.mkdir(parents=True, exist_ok=True)
    save_rrd(args.output_rrd)
    logging.info(f"Saved to {args.output_rrd}")


if __name__ == "__main__":
    main()
