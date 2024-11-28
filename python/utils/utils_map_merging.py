import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation/third_party/deep-image-retrieval'))

import argparse
from datetime import datetime
import logging
import numpy as np
import faiss

import vpr_models
from estimator import get_estimator, available_models
from estimator.utils import to_numpy
import matplotlib.pyplot as plt

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
	os.makedirs(os.path.join(out_dir, "seq"), exist_ok=True)
	os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
	# start_time = datetime.now()
	# log_dir = os.path.join(out_dir, f"outputs_{args.pose_estimation_method}", start_time.strftime("%Y-%m-%d_%H-%M-%S"))
	# setup_logging(log_dir, stdout_level="info")
	# logging.info(" ".join(sys.argv))
	# logging.info(f"Arguments: {args}")
	# logging.info(f"Testing with {args.pose_estimation_method} with image size {args.image_size}")
	# logging.info(f"The outputs are being saved in {log_dir}")
	# os.makedirs(os.path.join(log_dir, "preds"))
	# os.system(f"rm {os.path.join(out_dir, f'outputs_{args.pose_estimation_method}', 'latest')}")
	# os.system(f"ln -s {log_dir} {os.path.join(out_dir, f'outputs_{args.pose_estimation_method}', 'latest')}")
	# return log_dir
	return out_dir

# def initialize_vpr_model(method, backbone, descriptors_dimension, device):
# 	"""Initialize and return the model."""
# 	model = vpr_models.get_model(method, backbone, descriptors_dimension)
# 	return model.eval().to(device)

def initialize_pose_estimator(model, device):
	"""Initialize and return the model."""
	return get_estimator(model, device=device)

"""
Visualization
"""
def save_vis_coarse_loc(log_dir, db_submap, query_submap, query_submap_id, preds):
	db_images = [to_numpy(node.rgb_image.permute(1, 2, 0)) for _, node in db_submap.nodes.items()]
	query_images = [to_numpy(node.rgb_image.permute(1, 2, 0)) for _, node in query_submap.nodes.items()]
	fig, axes = plt.subplots(preds.shape[0], preds.shape[1]+1, figsize=(20, 2 * (preds.shape[1]+1)))
	for query_id in range(preds.shape[0]):
		axes[query_id, 0].imshow(query_images[query_id])
		axes[query_id, 0].set_title(f'Q{query_id + 1}')
		for i in range(preds.shape[1]):
			axes[query_id, i + 1].imshow(db_images[preds[query_id, i]])
			axes[query_id, i + 1].set_title(f'DB{preds[query_id, i] + 1}')
	plt.savefig(os.path.join(log_dir, f"preds/results_{query_submap_id}_coarse_loc.png"))

def save_vis_pose_graph(log_dir, db_submap, query_submap, query_submap_id, edges_nodeA_to_nodeB):
	"""
	Save visualization of graph-based map with nodes and edges.
	Plot the trajectory onto the X-Z plane.
	"""
	fig, ax = plt.subplots(figsize=(10, 10))
	# Plot submap
	for node_id, node in db_submap.nodes.items():
		ax.plot(node.trans_gt[0], node.trans_gt[2], 'bo', markersize=5)
		ax.text(node.trans_gt[0], node.trans_gt[2], f'DB{node_id}', fontsize=8, color='k')
		for edge in node.edges:
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[2], next_node.trans_gt[2]], 'k-')

	for node_id, node in query_submap.nodes.items():			
		ax.plot(node.trans_gt[0], node.trans_gt[2], 'go', markersize=5)
		ax.text(node.trans_gt[0], node.trans_gt[2], f'Q{node_id}', fontsize=8, color='k')		
		for edge in node.edges:
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[2], next_node.trans_gt[2]], 'k-')
	# Plot connections
	for edge in edges_nodeA_to_nodeB:
		nodeA, nodeB, _ = edge
		ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[2], nodeB.trans_gt[2]], 'r-')
	fig.tight_layout()
	ax.grid(ls='--', color='0.7')
	plt.axis('equal')
	plt.xlabel('X-axis'); plt.ylabel('Z-axis')
	plt.savefig(os.path.join(log_dir, f"preds/results_{query_submap_id}_coarse_loc_connection.png"))

def parse_arguments():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
											help="Resizing shape for images (WxH). If a single int is passed, set the"
											"smallest edge of all images to this value, while keeping aspect ratio")
	parser.add_argument("--num_submap", type=int, default=2, help="number of submaps for merging")
	
	parser.add_argument("--pose_estimation_method", type=str, default="master",
						help="master, duster")

	# parser.add_argument("--positive_dist_threshold", type=int, default=25,
	# 										help="distance (in meters) for a prediction to be considered a positive")
	# parser.add_argument("--vpr_method", type=str, default="cosplace",
	# 											choices=["netvlad", "apgem", "sfrs", "cosplace", "convap", "mixvpr", "eigenplaces", 
	# 													"eigenplaces-indoor", "anyloc", "salad", "salad-indoor", "cricavpr"],
	# 											help="_")
	# parser.add_argument("--backbone", type=str, default=None,
	# 										choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
	# 										help="_")
	# parser.add_argument("--descriptors_dimension", type=int, default=None, help="_")

	# parser.add_argument("--num_workers", type=int, default=4,
	# 										help="_")
	# parser.add_argument("--batch_size", type=int, default=4,
	# 										help="set to 1 if database images may have different resolution")
	# parser.add_argument("--log_dir", type=str, default="default", 
	# 				 help="experiment name, output logs will be saved under logs/log_dir")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
	# parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
	# 										help="values for recall (e.g. recall@1, recall@5)")
	# parser.add_argument("--no_labels", action="store_true",
	# 										help="set to true if you have no labels and just want to "
	# 										"do standard image retrieval given two folders of queries and DB")
	# parser.add_argument("--num_preds_to_save", type=int, default=0,
	# 										help="set != 0 if you want to save predictions for each query")
	# parser.add_argument("--save_only_wrong_preds", action="store_true",
	# 										help="set to true if you want to save predictions only for "
	# 										"wrongly predicted queries")
	# parser.add_argument("--save_descriptors", action="store_true",
	# 										help="set to True if you want to save the descriptors extracted by the model")
	args = parser.parse_args()
	
	# args.use_labels = not args.no_labels
	
	# if args.method == "netvlad":
	# 	if args.backbone not in [None, "VGG16"]:
	# 		raise ValueError("When using NetVLAD the backbone must be None or VGG16")
	# 	if args.descriptors_dimension not in [None, 4096, 32768]:
	# 		raise ValueError("When using NetVLAD the descriptors_dimension must be one of [None, 4096, 32768]")
	# 	if args.descriptors_dimension is None:
	# 		args.descriptors_dimension = 4096
			
	# elif args.method == "sfrs":
	# 	if args.backbone not in [None, "VGG16"]:
	# 		raise ValueError("When using SFRS the backbone must be None or VGG16")
	# 	if args.descriptors_dimension not in [None, 4096]:
	# 		raise ValueError("When using SFRS the descriptors_dimension must be one of [None, 4096]")
	# 	if args.descriptors_dimension is None:
	# 		args.descriptors_dimension = 4096
	
	# elif args.method == "cosplace":
	# 	if args.backbone is None:
	# 		args.backbone = "ResNet50"
	# 	if args.descriptors_dimension is None:
	# 		args.descriptors_dimension = 512
	# 	if args.backbone == "VGG16" and args.descriptors_dimension not in [64, 128, 256, 512]:
	# 		raise ValueError("When using CosPlace with VGG16 the descriptors_dimension must be in [64, 128, 256, 512]")
	# 	if args.backbone == "ResNet18" and args.descriptors_dimension not in [32, 64, 128, 256, 512]:
	# 		raise ValueError("When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]")
	# 	if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [32, 64, 128, 256, 512, 1024, 2048]:
	# 		raise ValueError(f"When using CosPlace with {args.backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]")
	
	# elif args.method == "convap":
	# 	if args.backbone is None:
	# 		args.backbone = "ResNet50"
	# 	if args.descriptors_dimension is None:
	# 		args.descriptors_dimension = 512
	# 	if args.backbone not in [None, "ResNet50"]:
	# 		raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
	# 	if args.descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
	# 		raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]")
	
	# elif args.method == "mixvpr":
	# 	if args.backbone is None:
	# 		args.backbone = "ResNet50"
	# 	if args.descriptors_dimension is None:
	# 		args.descriptors_dimension = 512
	# 	if args.backbone not in [None, "ResNet50"]:
	# 		raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
	# 	if args.descriptors_dimension not in [None, 128, 512, 4096]:
	# 		raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 128, 512, 4096]")
	
	# elif args.method == "eigenplaces":
	# 	if args.backbone is None:
	# 		args.backbone = "ResNet50"
	# 	if args.descriptors_dimension is None:
	# 		args.descriptors_dimension = 512
	# 	if args.backbone == "VGG16" and args.descriptors_dimension not in [512]:
	# 		raise ValueError("When using EigenPlaces with VGG16 the descriptors_dimension must be in [512]")
	# 	if args.backbone == "ResNet18" and args.descriptors_dimension not in [256, 512]:
	# 		raise ValueError("When using EigenPlaces with ResNet18 the descriptors_dimension must be in [256, 512]")
	# 	if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [128, 256, 512, 2048]:
	# 		raise ValueError(f"When using EigenPlaces with {args.backbone} the descriptors_dimension must be in [128, 256, 512, 2048]")
				
	# elif args.method == "eigenplaces-indoor":
	# 	args.backbone = "ResNet50"
	# 	args.descriptors_dimension = 2048
	
	# elif args.method == "apgem":
	# 	args.backbone = "Resnet101"
	# 	args.descriptors_dimension = 2048
	
	# elif args.method == "anyloc":
	# 	args.backbone = "DINOv2"
	# 	args.descriptors_dimension = 49152
	
	# elif args.method == "salad":
	# 	args.backbone = "DINOv2"
	# 	args.descriptors_dimension = 8448
			
	# elif args.method == "salad-indoor":
	# 	args.backbone = "Dinov2"
	# 	args.descriptors_dimension = 8448
	
	# elif args.method == "cricavpr":
	# 	args.backbone = "Dinov2"
	# 	args.descriptors_dimension = 10752
	
	# if args.image_size and len(args.image_size) > 2:
	# 	raise ValueError(f"The --image_size parameter can only take up to 2 values, but has received {len(args.image_size)}.")
	
	return args

