"""
Usage: 
python demo_loc.py \
--matcher master \
--config data/toy_example/loc_config.yaml \
--path_rgb_img0 data/toy_example/im2.jpg \
--path_rgb_img1 data/toy_example/im3.jpg \
--path_depth_img0 data/toy_example/im2_depth.png \
--path_depth_img1 data/toy_example/im3_depth.png \
--path_intrinsics0 data/toy_example/im2_intrinsics.txt \
--path_intrinsics1 data/toy_example/im3_intrinsics.txt
"""

import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map-free-reloc"))

from tqdm import tqdm
import torch
import argparse
import matplotlib
from pathlib import Path
import numpy as np

from matching import available_models, get_matcher
from matching.utils import to_numpy, get_image_pairs_paths

from utils.utils_image_matching_method import *
from utils.utils_image import load_rgb_image, load_depth_image
from utils.pose_solver import available_solvers, get_solver
from benchmark_rpe.rpe_default import cfg

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")

def setup_args():
    """Setup command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Matching Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="path to config file")
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

    parser.add_argument(
        "--path_rgb_img0", type=str, default=None, help="path to path_rgb_img0"
    )

    parser.add_argument(
        "--path_rgb_img1", type=str, default=None, help="path to path_rgb_img1"
    )

    parser.add_argument(
        "--path_depth_img0", type=str, default=None, help="path to path_depth_img0"
    )

    parser.add_argument(
        "--path_depth_img1", type=str, default=None, help="path to path_depth_img1"
    )

    # Path intrinsics for Mickey matcher (if not provided, we use defaults)
    parser.add_argument(
        "--path_intrinsics0", type=str, default=None, help="path to intrinsics0"
    )
    
    parser.add_argument(
        "--path_intrinsics1", type=str, default=None, help="path to intrinsics1"
    )

    # Pose solver selection
    parser.add_argument(
        "--pose_solver",
        type=str,
        nargs="+",
        default=None,
        choices=available_solvers,
    )
    return parser.parse_args()

def read_intrinsics(path_intrinsics, resize):
    """
    Reads the intrinsic camera parameters from a file and optionally resizes them.    
    
    Parameters
    ----------
    path_intrinsics : str
    resize : tuple or list of int, optional parameters should be scaled. If None, no resizing is performed.
    Returns
    -------
    numpy.ndarray: The intrinsic camera matrix after optional resizing.
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
    K0 = read_intrinsics(args.path_intrinsics0, image_size)
    K1 = read_intrinsics(args.path_intrinsics1, image_size)

    matcher = get_matcher(
        args.matcher, device=args.device, max_num_keypoints=args.n_kpts
    )
    if args.matcher == "mickey":
        matcher.resize = image_size
        matcher.K0 = torch.from_numpy(K0).unsqueeze(0).to(args.device)
        matcher.K1 = torch.from_numpy(K0).unsqueeze(0).to(args.device)

    """Setup Pose Solver"""
    cfg.merge_from_file(args.config)
    pose_solver_list = []
    for solver in args.pose_solver:
        pose_solver_list.append(get_solver(solver, cfg))

    """Load images and perform feature matching and pose estimation"""
    image0 = load_rgb_image(args.path_rgb_img0, resize=image_size)
    image1 = load_rgb_image(args.path_rgb_img1, resize=image_size)
    if args.path_depth_img0 is not None:
        depth0 = load_depth_image(args.path_depth_img0, depth_scale=0.001)
    else:
        depth0 = None
    if args.path_depth_img1 is not None:
        depth1 = load_depth_image(args.path_depth_img1, depth_scale=0.001)
    else:
        depth1 = None
    print(image0.shape, image1.shape, depth0.shape, depth1.shape)

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

    out_str = f"Paths: {str(args.path_rgb_img0), str(args.path_rgb_img1)}. Found {num_inliers} inliers after RANSAC. "
    if args.no_viz:
        viz_path = save_visualization(
            image0, image1, mkpts0, mkpts1, log_dir, 0, n_viz=50
        )
        out_str += f"Viz saved in {viz_path}. "
    print(out_str)

    """Perform Pose Estimation"""
    if args.matcher == "mickey":
        R, t = matcher.scene["R"].squeeze(0), matcher.scene["t"].squeeze(0)
        R, t = to_numpy(R), to_numpy(t)
        T_cam1_cam0 = np.eye(4)
        T_cam1_cam0[:3, :3], T_cam1_cam0[:3, 3] = R, t
        print(f'Mickey Solver:\n', T_cam1_cam0)
    elif args.matcher == "duster":
        im_poses = to_numpy(matcher.scene.get_im_poses())
        T_cam1_cam0 = (
            im_poses[0]
            if abs(np.sum(np.diag(im_poses[1])) - 4.0) < 1e-5
            else np.linalg.inv(im_poses[1])
        )
        print(f'Duster Solver:\n', T_cam1_cam0)
    for solver in tqdm(pose_solver_list):
        try:
            depth_img0 = to_numpy(depth0.squeeze(0))
            depth_img1 = to_numpy(depth1.squeeze(0))
            R, t, inliers = solver.estimate_pose(mkpts0, mkpts1, K0, K1, depth_img0, depth_img1)
            T_cam1_cam0 = np.eye(4)
            T_cam1_cam0[:3, :3], T_cam1_cam0[:3, 3] = R, t.reshape(3)
            print(f'{solver}:\nNumber of inliers:{inliers}\n', T_cam1_cam0)
        except Exception as e:
            print(f'Failed to estimate pose with {solver}:', e)

    """Visualization"""
    if args.matcher == "duster" and not args.no_viz:
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
