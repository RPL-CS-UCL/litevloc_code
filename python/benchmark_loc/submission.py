# [Computation]:
# python submission.py --config path/to/config --checkpoint path/to/checkpoint --o results/your_method --split val
# [Evaluation]:
# python -m mickey/benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/mickey_essentialmatrixmetric/submission.zip --split val

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import time
import torch
import numpy as np
from tqdm import tqdm

from transforms3d.quaternions import mat2quat

from pose_solver import available_solvers, get_solver
from matching import available_models, get_matcher
from matching.utils import to_numpy, get_image_pairs_paths

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map-free-reloc"))
from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device

@dataclass
class Pose:
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
    for data in tqdm(loader):
        try:
            data = data_to_model_device(data, matcher)
            rgb_img0, rgb_img1 = data['image0'], data['image1']
            rgb_img0 = rgb_img0.squeeze(0)
            rgb_img1 = rgb_img1.squeeze(0)

            K0, K1 = data['K_color0'], data['K_color1']
            if str_matcher == "mickey":
                matcher.K0, matcher.K1 = K0, K1

            """Image Matching"""
            start_time = time.time()
            matcher_result = matcher(rgb_img0, rgb_img1)
            matching_time = time.time() - start_time
            num_inliers, mkpts0, mkpts1 = (
                matcher_result["num_inliers"],
                matcher_result["inliers0"],
                matcher_result["inliers1"],
            )
            # print("Found {} matched keypoints".format(num_inliers))

            """Pose Estimation"""
            start_time = time.time()
            if str_matcher == "mickey":
                R, t = matcher.scene["R"].squeeze(0), matcher.scene["t"].squeeze(0)
                R, t = to_numpy(R), to_numpy(t)
                inliers = to_numpy(matcher.scene["inliers"].squeeze(0))[0]
            # elif str_matcher == "duster":
            #     im_poses = to_numpy(matcher.scene.get_im_poses())
            #     T = (im_poses[0]
            #         if abs(np.sum(np.diag(im_poses[1])) - 4.0) < 1e-5
            #         else np.linalg.inv(im_poses[1]))        
            #     R, t = T[:3, :3], T[:3, 3]
            #     inliers = to_numpy(matcher_result["num_inliers"])
            else:
                depth_img0 = to_numpy(data['depth0'].squeeze(0))
                depth_img1 = to_numpy(data['depth1'].squeeze(0))
                K0 = to_numpy(K0.squeeze(0))
                K1 = to_numpy(K1.squeeze(0))
                R, t, inliers = solver.estimate_pose(mkpts0, mkpts1, K0, K1, depth_img0, depth_img1)
            solver_time = time.time() - start_time

            running_time.append(matching_time + solver_time)

            """Save Results"""
            scene = data['scene_id'][0]
            query_img = data['pair_names'][1][0]
            # ignore frames without poses (e.g. not enough feature matches)
            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                continue

            # populate results_dict
            estimated_pose = Pose(image_name=query_img,
                                q=mat2quat(R).reshape(-1),
                                t=t.reshape(-1),
                                inliers=inliers)
            # print(data['T_0to1'].squeeze(0).cpu().detach().numpy()[:3, 3], t)
            results_dict[scene].append(estimated_pose)

            # if str_matcher == "duster":
            #     matcher.scene.show(cam_size=0.05)
        except Exception as e:
            scene = data['scene_id'][0]
            query_img = data['pair_names'][1][0]            
            estimated_pose = Pose(image_name=query_img, q=mat2quat(np.eye(3)), t=np.zeros(3), inliers=0.0)
            results_dict[scene].append(estimated_pose)
            tqdm.write(f"Error with {str_matcher}: {e}")    
            tqdm.write(f"(duster) May occur due to no overlapping regions or insufficient matching.")

    avg_runtime = np.mean(running_time)
    return results_dict, avg_runtime


def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, "w") as zip:
        for scene, poses in results_dict.items():
            poses_str = "\n".join((str(pose) for pose in poses))
            # print(poses_str)
            zip.writestr(f"pose_{scene}.txt", poses_str.encode("utf-8"))

def eval(args):
    # Load configs
    cfg.merge_from_file(args.config)

    # Create dataloader for different datasets
    if args.split == 'test':
        dataloader = DataModule(cfg).test_dataloader()
    elif args.split == 'val':
        cfg.TRAINING.BATCH_SIZE = 2
        cfg.TRAINING.NUM_WORKERS = 2
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
                runtime_str = f"{model}_{pose_solver}: {avg_runtime:.3f}s"
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
        "--image_size",
        type=int,
        default=None,
        nargs="+",
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="pass --no_viz to avoid saving visualizations",
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
