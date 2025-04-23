import os
import sys
import logging
from datetime import datetime
import argparse
import numpy as np
import matplotlib
from matching import available_models
from utils.pose_solver import available_solvers

sys.path.extend([
	os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation'),
	os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation/third_party/deep-image-retrieval')
])

if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

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
	tmp_dir = os.path.join(out_dir, f'outputs_{args.vpr_method}_{args.img_matcher}')
	log_dir = os.path.join(tmp_dir, f'{args.vpr_backbone}_' + start_time.strftime('%Y-%m-%d_%H-%M-%S'))
	setup_logging(log_dir, stdout_level="info")
	logging.info(" ".join(sys.argv))
	logging.info(f"Arguments: {args}")
	logging.info(f"Testing with {args.vpr_method} with a {args.vpr_backbone} backbone and descriptors dimension {args.vpr_descriptors_dimension}")
	logging.info(f"Testing with {args.img_matcher} with image size {args.image_size}")
	logging.info(f"The outputs are being saved in {log_dir}")
	os.makedirs(os.path.join(log_dir, 'preds'))
	os.system(f"rm {os.path.join(tmp_dir, 'latest')}")
	os.system(f"ln -s {log_dir} {os.path.join(tmp_dir, 'latest')}")
	
	return log_dir

def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Visual Localization Pipeline",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	""" 
	Common parameters
	"""
	parser.add_argument("--map_path", type=str, default="matterport3d", help="path to map_path")
	parser.add_argument("--query_data_path", type=str, default="out_general", help="path to query data for test")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
										help="Resizing shape for images (WxH). If a single int is passed, set the"
										"smallest edge of all images to this value, while keeping aspect ratio")
	parser.add_argument("--viz", action="store_true", help="pass --viz for visualization of immidiate results")
	parser.add_argument("--unit_type", action="store_true", help="depth images are encoded to uint16 (True) or float32 (False)")
	parser.add_argument('--depth_scale', type=float, default=0.001, help='0.001 or 1')
	parser.add_argument("--ros_rgb_img_type", type=str, default='raw', help="raw or compressed")
	parser.add_argument("--global_pos_threshold", type=float, default=20.0, help="Distance threshold to actively global localization")
	parser.add_argument("--min_master_conf_thre", type=float, default=0.0, help="Threshold for the confidence of the Mast3R matches")
	parser.add_argument("--min_kpts_inliers_thre", type=int, default=200, help="Number of keypoint inliers to consider image matching as valid")
	parser.add_argument("--min_solver_inliers_thre", type=int, default=150, help="Number of solver inliers to consider image matching as valid")
	"""
	Parameters for VPR methods
	"""
	parser.add_argument("--positive_dist_threshold", type=int, default=25,
											help="distance (in meters) for a prediction to be considered a positive")
	parser.add_argument("--vpr_method", type=str, default="cosplace",
											choices=["netvlad", "apgem", "sfrs", "cosplace", "convap", "mixvpr", "eigenplaces", 
													 "eigenplaces-indoor", "anyloc", "salad", "salad-indoor", "cricavpr"],
											help="_")
	parser.add_argument("--vpr_backbone", type=str, default=None,
											choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
											help="_")
	parser.add_argument("--vpr_descriptors_dimension", type=int, default=None,
											help="_")
	
	parser.add_argument("--num_workers", type=int, default=4,
											help="_")
	parser.add_argument("--batch_size", type=int, default=4,
											help="set to 1 if database images may have different resolution")
	parser.add_argument("--log_dir", type=str, default="default",
											help="experiment name, output logs will be saved under logs/log_dir")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
											help="_")
	parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
											help="values for recall (e.g. recall@1, recall@5)")
	parser.add_argument("--no_labels", action="store_true",
											help="set to true if you have no labels and just want to "
											"do standard image retrieval given two folders of queries and DB")
	parser.add_argument("--num_preds_to_save", type=int, default=0,
											help="set != 0 if you want to save predictions for each query")
	parser.add_argument("--save_only_wrong_preds", action="store_true",
											help="set to true if you want to save predictions only for "
											"wrongly predicted queries")
	parser.add_argument("--save_descriptors", action="store_true",
											help="set to True if you want to save the descriptors extracted by the model")
	
	"""
	Parameters for image matching
	"""
	parser.add_argument("--img_matcher", type=str, default="sift-lg", choices=available_models, help="choose your matcher")
	parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
	parser.add_argument("--save_img_matcher", action="store_true",
											help="set to True if you want to save image matching by the model")	

	"""
	Parameters for pose solver
	"""
	parser.add_argument("--pose_solver", type=str, default="pnp", choices=available_solvers)
	parser.add_argument("--config_pose_solver", type=str, default="matterport3d.yaml")

	"""
	Parse the argments
	"""
	args, unknown = parser.parse_known_args()
	args.use_labels = not args.no_labels
	
	if args.vpr_method == "netvlad":
			if args.vpr_backbone not in [None, "VGG16"]:
					raise ValueError("When using NetVLAD the backbone must be None or VGG16")
			if args.vpr_descriptors_dimension not in [None, 4096, 32768]:
					raise ValueError("When using NetVLAD the descriptors_dimension must be one of [None, 4096, 32768]")
			if args.vpr_descriptors_dimension is None:
					args.vpr_descriptors_dimension = 4096
			
	elif args.vpr_method == "sfrs":
			if args.vpr_backbone not in [None, "VGG16"]:
					raise ValueError("When using SFRS the backbone must be None or VGG16")
			if args.vpr_descriptors_dimension not in [None, 4096]:
					raise ValueError("When using SFRS the descriptors_dimension must be one of [None, 4096]")
			if args.vpr_descriptors_dimension is None:
					args.vpr_descriptors_dimension = 4096
	
	elif args.vpr_method == "cosplace":
			if args.vpr_backbone is None:
					args.vpr_backbone = "ResNet50"
			if args.vpr_descriptors_dimension is None:
					args.vpr_descriptors_dimension = 512
			if args.vpr_backbone == "VGG16" and args.vpr_descriptors_dimension not in [64, 128, 256, 512]:
					raise ValueError("When using CosPlace with VGG16 the descriptors_dimension must be in [64, 128, 256, 512]")
			if args.vpr_backbone == "ResNet18" and args.vpr_descriptors_dimension not in [32, 64, 128, 256, 512]:
					raise ValueError("When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]")
			if args.vpr_backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.vpr_descriptors_dimension not in [32, 64, 128, 256, 512, 1024, 2048]:
					raise ValueError(f"When using CosPlace with {args.vpr_backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]")
	
	elif args.vpr_method == "convap":
			if args.vpr_backbone is None:
					args.vpr_backbone = "ResNet50"
			if args.vpr_descriptors_dimension is None:
					args.vpr_descriptors_dimension = 512
			if args.vpr_backbone not in [None, "ResNet50"]:
					raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
			if args.vpr_descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
					raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]")
	
	elif args.vpr_method == "mixvpr":
			if args.vpr_backbone is None:
					args.vpr_backbone = "ResNet50"
			if args.vpr_descriptors_dimension is None:
					args.vpr_descriptors_dimension = 512
			if args.vpr_backbone not in [None, "ResNet50"]:
					raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
			if args.vpr_descriptors_dimension not in [None, 128, 512, 4096]:
					raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 128, 512, 4096]")
	
	elif args.vpr_method == "eigenplaces":
			if args.vpr_backbone is None:
					args.vpr_backbone = "ResNet50"
			if args.vpr_descriptors_dimension is None:
					args.vpr_descriptors_dimension = 512
			if args.vpr_backbone == "VGG16" and args.vpr_descriptors_dimension not in [512]:
					raise ValueError("When using EigenPlaces with VGG16 the descriptors_dimension must be in [512]")
			if args.vpr_backbone == "ResNet18" and args.vpr_descriptors_dimension not in [256, 512]:
					raise ValueError("When using EigenPlaces with ResNet18 the descriptors_dimension must be in [256, 512]")
			if args.vpr_backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.vpr_descriptors_dimension not in [128, 256, 512, 2048]:
					raise ValueError(f"When using EigenPlaces with {args.vpr_backbone} the descriptors_dimension must be in [128, 256, 512, 2048]")
					
	elif args.vpr_method == "eigenplaces-indoor":
			args.vpr_backbone = "ResNet50"
			args.vpr_descriptors_dimension = 2048
	
	elif args.vpr_method == "apgem":
			args.vpr_backbone = "Resnet101"
			args.vpr_descriptors_dimension = 2048
	
	elif args.vpr_method == "anyloc":
			args.vpr_backbone = "DINOv2"
			args.vpr_descriptors_dimension = 49152
	
	elif args.vpr_method == "salad":
			args.vpr_backbone = "DINOv2"
			args.vpr_descriptors_dimension = 8448
			
	elif args.vpr_method == "salad-indoor":
			args.vpr_backbone = "Dinov2"
			args.vpr_descriptors_dimension = 8448
	
	elif args.vpr_method == "cricavpr":
			args.vpr_backbone = "Dinov2"
			args.vpr_descriptors_dimension = 10752
	
	if args.image_size is not None and len(args.image_size) > 2:
			raise ValueError(f"The --image_size parameter can only take up to 2 values, but has received {len(args.image_size)}.")
	
	if args.unit_type and abs(args.depth_scale - 1.0) > 1e-3:
			raise ValueError(f"The depth scale should be 1.0 not {args.depth_scale} since unit_type is True.")

	return args

def save_descriptors(log_dir, descriptors):
	"""Save descriptors to files."""
	logging.debug(f"Saving the descriptors in {log_dir}")
	np.savetxt(
		os.path.join(log_dir, f'database_descriptors.txt'), 
		descriptors, 
		fmt='%s ' + '%.9f ' * descriptors.shape[1]
	)

if __name__ == "__main__":
	args = parse_arguments()
	setup_log_environment('/tmp/', args)