"""
Usage: python test_image_matching_method.py \
--matcher duster \
--input /Titan/code/robohike_ws/src/topo_loc/python/test/logs/match_pairs.txt \
--image_sizesd 288 512
"""

import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import torch
import argparse
import matplotlib
from pathlib import Path
import numpy as np

from matching import available_models, get_matcher
from matching.utils import to_numpy, get_image_pairs_paths

from utils.utils_image_matching_method import *
from utils.utils_image import load_rgb_image, load_depth_image
from pose_solver import *

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")


def setup_args():
    """Setup command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Matching Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        nargs="+",
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="pass --no_viz to avoid saving visualizations",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="assets/example_pairs",
        help="path to either (1) dir with dirs with pairs or (2) txt file with two img paths per line",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="path where outputs are saved"
    )

    # Path intrinsics for Mickey matcher (if not provided, we use defaults)
    parser.add_argument(
        "--path_intrinsics", type=str, default=None, help="path to intrinsics"
    )

    # Pose solver selection
    parser.add_argument(
        "--pose_solver",
        type=str,
        default="EssentialMatrix",
        choices=["EssentialMatrix", "EssentialMatrixMetric", "Procrustes", "PNP"],
    )

    return parser.parse_args()


def main(args):
    """Setup Logging"""
    image_size = args.image_size
    log_dir = setup_log_environment(args.out_dir, args)

    """Setup Matcher"""
    matcher = get_matcher(
        args.matcher, device=args.device, max_num_keypoints=args.n_kpts
    )
    if args.matcher == "mickey":
        matcher.resize = image_size
        matcher.path_intrinsics = args.path_intrinsics

    """Setup Pose Solver"""
    # cfg.EMAT_RANSAC.PIX_THRESHOLD = 100.0
    if args.pose_solver == "EssentialMatrix":
        pose_solver = EssentialMatrixSolver()
    elif args.pose_solver == "EssentialMatrixMetric":
        pose_solver = EssentialMatrixMetricSolver()
    elif args.pose_solver == "Procrustes":
        pose_solver = ProcrustesSolver()
    elif args.pose_solver == "PNP":
        pose_solver = PnPSolver()
    else:
        raise NotImplementedError("Invalid pose solver")

    """Load images and perform feature matching and pose estimation"""
    pairs_of_paths = get_image_pairs_paths(args.input)
    for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
        image0 = load_rgb_image(img0_path, resize=image_size)
        image1 = load_rgb_image(img1_path, resize=image_size)

        """Perform Feature Matching"""
        start_time = time.time()
        result = matcher(image0, image1)
        print("Matching costs time: {:3f}s".format(time.time() - start_time))
        num_inliers, H, mkpts0, mkpts1 = (
            result["num_inliers"],
            result["H"],
            result["inliers0"],
            result["inliers1"],
        )
        print("Found {} matched keypoints".format(num_inliers))

        out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {num_inliers} inliers after RANSAC. "
        if not args.no_viz:
            viz_path = save_visualization(
                image0, image1, mkpts0, mkpts1, log_dir, i, n_viz=100
            )
            out_str += f"Viz saved in {viz_path}. "
        print(out_str)

        """Perform Pose Estimation"""
        if args.matcher == "mickey":
            R, t = matcher.scene["R"], matcher.scene["t"]
            R, t = to_numpy(R), to_numpy(t)
            print(R, t)

        """Visualization"""
        if args.matcher == "duster" and args.no_viz:
            scene = matcher.scene
            scene.show(cam_size=0.05)


if __name__ == "__main__":
    args = setup_args()
    if args.out_dir is None:
        args.out_dir = Path(f"outputs_image_matching")
    args.out_dir = Path(args.out_dir)
    if args.image_size and len(args.image_size) > 2:
        raise ValueError(
            f"The --image_size parameter can only take up to 2 values, but has received {len(args.image_size)}."
        )
    main(args)
