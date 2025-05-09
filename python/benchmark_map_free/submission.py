# [Computation]:
# python submission.py --config ../config/dataset/mapfree.yaml --models master --pose_solver essentialmatrixmetricmean --out_dir /Titan/dataset/data_mapfree/results --split val
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetricmean/submission.zip --dataset_path /Titan/dataset/data_mapfree --split val --log error

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile
import time
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat

from utils.pose_solver import available_solvers, get_solver
from benchmark_rpe.rpe_default import cfg
from utils.utils_image_matching_method import save_visualization
from utils.utils_geom import correct_intrinsic_scale

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map_free_reloc"))
from lib.datasets.datamodules import DataModule
from lib.utils.data import data_to_model_device

# Matching framework imports
from matching import available_models, get_matcher
from matching.utils import to_numpy

@dataclass
class PoseResult:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {"float": lambda v: f"{v:.6f}"}
        max_line_width = 1000
        q_str = np.array2string(
            self.q, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        t_str = np.array2string(
            self.t, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        return f"{self.image_name} {q_str} {t_str} {self.inliers}"

def predict(loader, matcher, solver, str_matcher, str_solver):
    results_dict = defaultdict(list)
    running_time = []
    save_indice = 0
    for data in tqdm(loader):
        try:
            # NOTE(gogojjh): the original data loader: rgb images are resize, depth images are not resized
            data = data_to_model_device(data, matcher)
            rgb_img0, rgb_img1 = data['image0'], data['image1']
            rgb_img0 = rgb_img0.squeeze(0)
            rgb_img1 = rgb_img1.squeeze(0)
            K0, K1 = data['K_color0'], data['K_color1']
            if str_matcher == "mickey": matcher.K0, matcher.K1 = K0, K1

            """Image Matching"""
            start_time = time.time()
            matcher_result = matcher(rgb_img0, rgb_img1)
            matching_time = time.time() - start_time
            num_inliers, mkpts0, mkpts1 = (
                matcher_result["num_inliers"],
                matcher_result["inlier_kpts0"],
                matcher_result["inlier_kpts1"],
            )

            """Pose Estimation"""
            start_time = time.time()
            if str_matcher == "mickey":
                R, t = matcher.scene["R"].squeeze(0), matcher.scene["t"].squeeze(0)
                R, t = to_numpy(R), to_numpy(t)
                inliers_solver = to_numpy(matcher.scene["inliers"].squeeze(0))[0]
            else:
                depth_img0 = to_numpy(data['depth0'].squeeze(0))
                depth_img1 = to_numpy(data['depth1'].squeeze(0))
                if cfg.DATASET.MAX_DEPTH is not None:
                    depth_img0[depth_img0 > cfg.DATASET.MAX_DEPTH] = 0.0
                    depth_img1[depth_img1 > cfg.DATASET.MAX_DEPTH] = 0.0
                K0 = to_numpy(K0.squeeze(0))
                K1 = to_numpy(K1.squeeze(0))
                ori_w, ori_h = depth_img0.shape[1], depth_img0.shape[0]
                K0_raw = correct_intrinsic_scale(K0, ori_w / rgb_img0.shape[2], ori_h / rgb_img0.shape[1])
                K1_raw = correct_intrinsic_scale(K1, ori_w / rgb_img1.shape[2], ori_h / rgb_img1.shape[1])
                mkpts0_raw = mkpts0 * [ori_w / rgb_img0.shape[2], ori_h / rgb_img0.shape[1]]
                mkpts1_raw = mkpts1 * [ori_w / rgb_img1.shape[2], ori_h / rgb_img1.shape[1]]
                """Definition of solver output"""
                # R01 (numpy.ndarray): Estimated rotation matrix. Shape: [3, 3] that rotate depth_img1 to depth_img0.
                # t01 (numpy.ndarray): Estimated translation vector. Shape: [3, 1] that translate depth_img1 to depth_img0.
                # inliers_solver (int): Number of inliers used in the final pose estimation.
                R01, t01, inliers_solver = solver.estimate_pose(
                    mkpts1_raw, mkpts0_raw, K1_raw, K0_raw, depth_img1, depth_img0
                )
                T01 = np.eye(4); T01[:3, :3] = R01; T01[:3,  3] = t01.reshape(3)
                T10 = np.linalg.inv(T01); R = T10[:3, :3]; t = T10[:3,  3].reshape(3, 1)
            solver_time = time.time() - start_time           
            running_time.append(matching_time + solver_time)

            """Save Results"""
            query_img = data['pair_names'][1][0]
            scene = data['scene_id'][0]
            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                raise ValueError("Estimated pose is NaN or infinite.")

            # populate results_dict
            estimated_pose = PoseResult(
                image_name=query_img,
                q=mat2quat(R).reshape(-1),
                t=t.reshape(-1),
                inliers=num_inliers
            )
            results_dict[scene].append(estimated_pose)

            if args.debug:
                print(t.T)
                if num_inliers < 100:
                    print(f"Inliers number < 100: {num_inliers} at {data['scene_id'][0]}/{data['pair_names']}")
                out_match_dir = Path(os.path.join(args.out_dir, f"{str_matcher}_{str_solver}"))
                out_match_dir.mkdir(parents=True, exist_ok=True)
                Path(out_match_dir / "preds").mkdir(parents=True, exist_ok=True)
                text = f"{len(mkpts1)} matches: {scene}-{query_img.split('/')[1]}" # "N matches: s00000-frame_000000.jpg"
                save_visualization(rgb_img0, rgb_img1, mkpts0, mkpts1, out_match_dir, save_indice, n_viz=30, line_width=0.6, text=text)
                save_indice += 1
                if str_matcher == "duster" and args.viz: matcher.scene.show(cam_size=0.05)
        except Exception as e:
            scene = data['scene_id'][0]
            query_img = data['pair_names'][1][0]
            tqdm.write(f"Error with {str_matcher}: {e}")
            tqdm.write(f"(duster) May occur due to no overlapping regions or insufficient matching at {scene}/{query_img}.")

    avg_runtime = np.mean(running_time)
    return results_dict, avg_runtime

def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, "w") as zip:
        for scene, poses in results_dict.items():
            poses_str = "\n".join((str(pose) for pose in poses))
            zip.writestr(f"pose_{scene}.txt", poses_str.encode("utf-8"))

def eval(args):
    # Load configs
    cfg.merge_from_file(args.config)

    # Create dataloader for different datasets
    if args.split == 'test':
        dataloader = DataModule(cfg).test_dataloader()
    elif args.split == 'val':
        cfg.TRAINING.BATCH_SIZE = 1
        cfg.TRAINING.NUM_WORKERS = 1
        dataloader = DataModule(cfg).val_dataloader()
    else:
        raise NotImplemented(f'Invalid split: {args.split}')

    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "runtime_results.txt", "w") as f:
        for model in args.models:
            matcher = get_matcher(model, device=args.device)
            for pose_solver in args.pose_solvers:
                print(f"Running Image Matching: {model} with Pose Solver: {pose_solver}")
                solver = get_solver(pose_solver, cfg)
                results_dict, avg_runtime = predict(dataloader, matcher, solver, model, pose_solver)

                # Save runtimes to txt
                runtime_str = f"(image_matching + pose_solver) {model}_{pose_solver}: {avg_runtime:.3f}s"
                f.write(runtime_str + "\n")
                tqdm.write(runtime_str)

                # Save predictions to txt per scene within zip
                log_dir = Path(os.path.join(output_root, f"{model}_{pose_solver}"))
                log_dir.mkdir(parents=True, exist_ok=True)
                save_submission(results_dict, log_dir / "submission.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default="all",
        choices=available_models
    )
    parser.add_argument(
        "--pose_solvers",
        type=str,
        nargs="+",
        default="all",
        choices=available_solvers,
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="pass --viz to avoid saving visualizations",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="pass --debug to visualize intermediate results",
    )    
    parser.add_argument(
        "--out_dir", type=str, default=None, help="path where outputs are saved"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="number of interations to run benchmark and average over",
    )    
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to use for evaluation. Choose from test or val. Default: test",
    )
    args = parser.parse_args()
    if args.models == "all":
        args.models = available_models
    if args.pose_solvers == "all":
        args.pose_solvers = available_solvers
    eval(args)
