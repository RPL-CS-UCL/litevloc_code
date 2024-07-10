import os
import sys
from datetime import datetime
import logging

import torch

from matching import viz2d, get_matcher

def setup_logging(log_dir, stdout_level='info'):
    os.makedirs(log_dir, exist_ok=True)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, stdout_level.upper(), 'INFO'),
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'info.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def setup_log_environment(out_dir, args):
    """Setup logging and directories."""
    os.makedirs(out_dir, exist_ok=True)
    start_time = datetime.now()
    log_dir = os.path.join(out_dir, f'outputs_{args.matcher}', start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    setup_logging(log_dir, stdout_level="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"Testing with {args.matcher} with image size {args.image_size}")
    logging.info(f"The outputs are being saved in {log_dir}")
    os.makedirs(os.path.join(log_dir, 'preds'))
    return log_dir

def save_visualization(image0, image1, mkpts0, mkpts1, out_dir, index, n_viz=1):
    """Save visualization of the matching results."""
    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(mkpts0[::n_viz], mkpts1[::n_viz], color="lime", lw=0.2)
    viz2d.add_text(0, f"{len(mkpts1)} matches", fs=20)
    viz_path = os.path.join(out_dir, "preds", f"{index:06d}.jpg")
    viz2d.save_plot(viz_path)
    return viz_path

def save_output(result, img0_path, img1_path, matcher_name, n_kpts, im_size, out_dir, index):
    """Save the output data to a file."""
    dict_path = os.path.join(out_dir, "preds", f"{index:06d}.torch")
    output_dict = {
        "num_inliers": result["num_inliers"],
        "H": result["H"],
        "mkpts0": result["inliers0"],
        "mkpts1": result["inliers1"],
        "img0_path": img0_path,
        "img1_path": img1_path,
        "matcher": matcher_name,
        "n_kpts": n_kpts,
        "im_size": im_size,
    }
    torch.save(output_dict, dict_path)
    return dict_path

def initialize_matcher(matcher, device, n_kpts):
    """Initialize the matcher with provided arguments."""
    return get_matcher(matcher, device=device, max_num_keypoints=n_kpts)
