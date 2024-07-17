import os
import sys
import argparse
from datetime import datetime
import logging
import numpy as np

import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
	parser.add_argument('--depth_scale', type=float, default=0.001, help='habitat: 0.039, anymal: 0.001')
	parser.add_argument('--min_depth_pro', type=float, default=0.1, help='pixels are processed only if depth > min_depth_pro')
	parser.add_argument('--max_depth_pro', type=float, default=5.5, help='pixels are processed only if depth < min_depth_pro')
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
	os.makedirs(os.path.join(log_dir, 'preds_depthmap'))
	os.system(f"rm {os.path.join(out_dir, f'outputs_{args.matcher}', 'latest')}")
	os.system(f"ln -s {log_dir} {os.path.join(out_dir, f'outputs_{args.matcher}', 'latest')}")
	return log_dir

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

def compute_scale_factor(A, B):
	"""
	Compute the scale factor s using the provided equation with a robust M-estimator, remove outliers
	
	Args:
			A (np.ndarray): Reference matrix (depth_image1).
			B (np.ndarray): Matrix to be scaled (depth_image2).
	
	Returns:
			float: Computed scale factor.
	"""
	def huber_loss(residual, delta=0.1):
		"""
		Huber loss function.
		
		Args:
				residual (np.ndarray): Residuals.
				delta (float): Delta parameter for Huber loss.
		
		Returns:
				float: Huber loss value.
		"""
		return np.where(np.abs(residual) <= delta, 0.5 * residual**2, delta * (np.abs(residual) - 0.5 * delta))

	def objective_function(s):
		"""
		Objective function to minimize.
		
		Args:
				s (float): Scale factor.
		
		Returns:
				float: Sum of Huber loss for residuals.
		"""
		residual = A - s * B
		return np.sum(huber_loss(residual))

	result = minimize(objective_function, x0=1.0)
	return result.x[0]

def compute_residual_matrix(A, B, s):
	def huber_loss(residual, delta=1.0):
		"""
		Huber loss function.
		
		Args:
				residual (np.ndarray): Residuals.
				delta (float): Delta parameter for Huber loss.
		
		Returns:
				float: Huber loss value.
		"""
		return np.where(np.abs(residual) <= delta, 0.5 * residual**2, delta * (np.abs(residual) - 0.5 * delta))
	return huber_loss(A - s * B)

def plot_images(image1, image2, title1="Image 1", title2="Image 2", save_path=None):
	"""
	Plot two images side by side with colorbars.
	
	Parameters:
	image1 (numpy.ndarray): The first image.
	image2 (numpy.ndarray): The second image.
	title1 (str): Title for the first image.
	title2 (str): Title for the second image.
	"""
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))

	im1 = axes[0].imshow(image1, cmap='viridis')
	axes[0].set_title(title1)
	axes[0].axis('off')
	fig.colorbar(im1, ax=axes[0])

	im2 = axes[1].imshow(image2, cmap='viridis', vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
	axes[1].set_title(title2)
	axes[1].axis('off')
	fig.colorbar(im2, ax=axes[1])

	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()

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
		print(np.argmax(trans_e))
		f.write(out_str)

def save_rgb_depth_images(rgb_image: np.array, depth_image: np.array, 
									        rgb_path: str, depth_path: str):
	from PIL import Image
	rgb_image = rgb_image.astype(np.uint8)      # HxWx3
	pil_image = Image.fromarray(rgb_image)
	pil_image.save(rgb_path)

	depth_image = depth_image.astype(np.uint16) # HxWx1
	pil_image = Image.fromarray(depth_image)
	pil_image.save(depth_path)	

def save_image(image: np.array, save_path: str):
	from PIL import Image
	image = image.astype(np.uint8)
	pil_image = Image.fromarray(image)
	pil_image.save(save_path)