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

import yaml

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
        "--out_dir", type=str, default=None, help="path where outputs are saved"
    )

    parser.add_argument("--img0", type=str, default=None, help="path to img0")
    parser.add_argument("--img1", type=str, default=None, help="path to img1")

    # Path intrinsics for Mickey matcher (if not provided, we use defaults)
    parser.add_argument(
        "--intrinsics0", type=str, default=None, help="path to intrinsics0"
    )
    parser.add_argument(
        "--intrinsics1", type=str, default=None, help="path to intrinsics1"
    )

    # Pose solver selection
    parser.add_argument(
        "--pose_solver",
        type=str,
        default=None,
        choices=["EssentialMatrix", "EssentialMatrixMetric", "Procrustes", "PNP"],
    )

    # Configure files
    parser.add_argument(
        "--config_file", type=str, default=None, help="path to config file"
    )

    return parser.parse_args()


def read_intrinsics(path_intrinsics, resize):
    """
    Reads the intrinsic camera parameters from a file and optionally resizes them.

    Parameters
    ----------
    path_intrinsics : str
        Path to the file containing the intrinsic parameters.
    resize : tuple or list of int, optional
        The new size (height, width) to which the intrinsic parameters should be scaled.
        If None, no resizing is performed.

    Returns
    -------
    numpy.ndarray
        The intrinsic camera matrix after optional resizing.
    """

    def correct_intrinsic_scale(K, scale_x, scale_y):
        """Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
        the new coordinates x' = scale_x * x; y' = scale_y * y
        Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        """
        transform = torch.eye(3)
        transform[0, 0] = scale_x
        transform[0, 2] = scale_x / 2 - 0.5
        transform[1, 1] = scale_y
        transform[1, 2] = scale_y / 2 - 0.5
        Kprime = transform @ K
        return Kprime

    with Path(path_intrinsics).open("r") as f:
        for line in f.readlines():
            fx, fy, cx, cy, W, H = map(float, line.split())
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            if resize is not None:
                K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H).numpy()
    return K


def main(args):
    """Setup Logging"""
    image_size = args.image_size
    log_dir = setup_log_environment(args.out_dir, args)

    """Setup Matcher"""
    img0_path, img1_path = args.img0, args.img1
    K0 = read_intrinsics(args.intrinsics0, image_size)
    K1 = read_intrinsics(args.intrinsics1, image_size)

    matcher = get_matcher(
        args.matcher, device=args.device, max_num_keypoints=args.n_kpts
    )
    if args.matcher == "mickey":
        matcher.resize = image_size
        matcher.K0 = K0
        matcher.K1 = K1

    """Setup Pose Solver"""
    with open(args.config_file, "r") as file:
        cfg = yaml.safe_load(file)
    if args.pose_solver == "EssentialMatrix":
        pose_solver = EssentialMatrixSolver(cfg)
    elif args.pose_solver == "EssentialMatrixMetric":
        pose_solver = EssentialMatrixMetricSolver(cfg)
    elif args.pose_solver == "Procrustes":
        pose_solver = ProcrustesSolver(cfg)
    elif args.pose_solver == "PNP":
        pose_solver = PnPSolver(cfg)
    else:
        raise NotImplementedError("Invalid pose solver")

    """Load images and perform feature matching and pose estimation"""
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
            image0, image1, mkpts0, mkpts1, log_dir, 0, n_viz=100
        )
        out_str += f"Viz saved in {viz_path}. "
    print(out_str)

    """Perform Pose Estimation"""
    if args.matcher == "mickey":
        R, t = matcher.scene["R"].squeeze(0), matcher.scene["t"].squeeze(0).T
        R, t = to_numpy(R), to_numpy(t)
        print(R, "\n", t)
    else:
        R, t, inliers = pose_solver.estimate_pose(mkpts0, mkpts1, K0, K1)
        t = t.reshape((3, 1))
        print(R, "\n", t)

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
