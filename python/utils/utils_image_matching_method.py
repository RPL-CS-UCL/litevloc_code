import os
import sys
import argparse
from datetime import datetime
import logging
import numpy as np

import torch

from matching import viz2d, get_matcher, available_models

def parse_arguments():
	"""Setup command-line arguments."""
	parser = argparse.ArgumentParser(description="Batch Image Matching Test",
																		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	parser.add_argument("--matcher", type=str, default="sift-lg", choices=available_models, help="choose your matcher")
	parser.add_argument("--image_size", type=int, default=512, nargs="+",
											help="Resizing shape for images (HxW). If a single int is passed, set the"
											"smallest edge of all images to this value, while keeping aspect ratio")
	parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
	parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
	parser.add_argument("--no_viz", action="store_true", help="pass --no_viz to avoid saving visualizations")
	parser.add_argument("--sample_map", type=int, default=1, help="sample of map")
	parser.add_argument("--sample_obs", type=int, default=1, help="sample of observation")
	parser.add_argument('--depth_scale', type=float, default='0.001', help='habitat: 0.039, anymal: 0.001')
	args = parser.parse_args()
	return args

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

def save_error(rot_e, trans_e, out_dir):
	err_path = os.path.join(out_dir, 'rot_trans_error.txt')
	with open(err_path, 'w') as f:
		out_str  = f'Rotation Error [degree]:\n'
		for e in rot_e: out_str += f'{e:.5f}, '
		out_str += f'\n'

		out_str += f'Translation Error [m]:\n'
		for e in trans_e: out_str += f'{e:.5f}, '
		out_str += f'\n'

		out_str += f'Mean, STD, Median, Min, Max Rotation Error [degree]:\n'
		out_str += f'{np.mean(rot_e):.3f}, {np.std(rot_e):.3f}, {np.median(rot_e):.3f}, {np.min(rot_e):.3f}, {np.max(rot_e):.3f}\n'
		out_str += f'Mean, STD, Median, Min, Max Translation Error [m]:\n'
		out_str += f'{np.mean(trans_e):.3f}, {np.std(trans_e):.3f}, {np.median(trans_e):.3f}, {np.min(trans_e):.3f}, {np.max(trans_e):.3f}\n'
		print(out_str)
		f.write(out_str)

def initialize_matcher(matcher, device, n_kpts):
	"""Initialize the matcher with provided arguments."""
	return get_matcher(matcher, device=device, max_num_keypoints=n_kpts)

def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)

def save_input_images(rgb_image: np.array, depth_image: np.array, 
									     rgb_path: str, depth_path: str):
	from PIL import Image
	rgb_image = np.transpose(rgb_image.astype(np.uint8), (1, 2, 0))      # 3xHXW -> HxWx3
	pil_image = Image.fromarray(rgb_image)
	pil_image.save(rgb_path)
	depth_image = np.transpose(depth_image.astype(np.uint16), (1, 2, 0)) # 1xHXW -> HxWx1
	depth_image = np.squeeze(depth_image, axis=2)
	pil_image = Image.fromarray(depth_image)
	pil_image.save(depth_path)	

def save_duster_images(rgb_image: np.array, depth_image: np.array, 
									     rgb_path: str, depth_path: str):
	from PIL import Image
	rgb_image = rgb_image.astype(np.uint8)      # HxWx3
	pil_image = Image.fromarray(rgb_image)
	pil_image.save(rgb_path)
	depth_image = depth_image.astype(np.uint16) # HxWx1
	pil_image = Image.fromarray(depth_image)
	pil_image.save(depth_path)	
