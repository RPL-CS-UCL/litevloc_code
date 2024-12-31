import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation/third_party/deep-image-retrieval'))

import argparse
# from datetime import datetime
import logging
import numpy as np

from estimator import get_estimator, available_models
from estimator.utils import to_numpy
import matplotlib.pyplot as plt

import pycpptools.src.python.utils_math as pytool_math

RMSE_THRESHOLD = 3.0
VPR_MATCH_THRESHOLD = 0.90
REFINE_EDGE_SCORE_THRESHOLD = 10.0 # threshold to select good refinement: out-of-range image, wrong coarse localization

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
def save_vis_vpr(log_dir, db_submap, query_submap, query_submap_id, preds, suffix=''):
	db_images = [to_numpy(node.rgb_image.permute(1, 2, 0)) for _, node in db_submap.nodes.items()]
	query_images = [to_numpy(node.rgb_image.permute(1, 2, 0)) for _, node in query_submap.nodes.items()]
	fig, axes = plt.subplots(preds.shape[0], preds.shape[1]+1, figsize=(20, 2 * (preds.shape[1]+1)))
	for query_id in range(preds.shape[0]):
		axes[query_id, 0].imshow(query_images[query_id])
		axes[query_id, 0].set_title(f'Q{query_id}')
		for i in range(preds.shape[1]):
			axes[query_id, i + 1].imshow(db_images[preds[query_id, i]])
			axes[query_id, i + 1].set_title(f'DB{preds[query_id, i]}')
	if suffix == '':
		plt.savefig(os.path.join(log_dir, f"preds/results_{query_submap_id}_vpr.png"))
	else:
		plt.savefig(os.path.join(log_dir, f"preds/results_{suffix}_{query_submap_id}_vpr.png"))

def save_vis_pose_graph(log_dir, db_submap, query_submap, query_submap_id, edges_nodeA_to_nodeB, suffix=''):
	"""
	Save visualization of graph-based map with nodes and edges.
	Plot the trajectory onto the X-Z plane.
	"""
	fig, ax = plt.subplots(figsize=(10, 10))
	
	# Plot submap
	for node_id, node in db_submap.nodes.items():
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'ko', markersize=5)
		# ax.text(node.trans_gt[0], node.trans_gt[1], f'DB{node_id}', fontsize=12, color='k')
		for edge in node.edges:
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)

	for node_id, node in query_submap.nodes.items():			
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
		ax.text(node.trans_gt[0], node.trans_gt[1], f'Q{node_id}', fontsize=12, color='k')		
		for edge in node.edges:
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)
	
	# Plot connections
	succ_cnt = 0
	for edge in edges_nodeA_to_nodeB:
		nodeA, nodeB, T_rel, prob = edge
		# Identify correct and wrong connections
		if 'coarse' in suffix:
			dis_tsl, dis_angle = \
				pytool_math.tools_eigen.compute_relative_dis(nodeA.trans_gt, nodeA.quat_gt, nodeB.trans_gt, nodeB.quat_gt)
			if dis_tsl < 10.0:
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'g-', linewidth=4)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={prob:.2f}', fontsize=12, color='k')
				succ_cnt += 1
			else:
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'r-', linewidth=4)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={prob:.2f}', fontsize=12, color='k')
				print(f"Wrong Connection: Query {nodeB.id} - DB {nodeA.id} with distance {dis_tsl:.2f}m")
		elif 'refine' in suffix:
			T_nodeA = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans_gt, nodeA.quat_gt)
			T_nodeB = pytool_math.tools_eigen.convert_vec_to_matrix(nodeB.trans_gt, nodeB.quat_gt)
			T_rel_gt = np.linalg.inv(T_nodeA) @ T_nodeB
			dis_tsl, dis_angle = \
				pytool_math.tools_eigen.compute_relative_dis_TF(T_rel, T_rel_gt)
			if dis_tsl < 1.0 and dis_angle < 45.0:
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'g-', linewidth=4)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={prob:.2f}', fontsize=12, color='k')
				succ_cnt += 1
			else:
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'r-', linewidth=4)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={prob:.2f}', fontsize=12, color='k')
	
	ax.grid(ls='--', color='0.7')
	plt.axis('equal')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title(f"Pose Graph with {succ_cnt}/{len(edges_nodeA_to_nodeB)} Connected Edges")
	if suffix == '':
		plt.savefig(os.path.join(log_dir, f"preds/results_{query_submap_id}_posegraph.png"))
	else:
		plt.savefig(os.path.join(log_dir, f"preds/results_{suffix}_{query_submap_id}_posegraph.png"))

def save_query_result(log_dir, query_result_info, query_submap_id):
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	for i in range(query_result_info.shape[0]):
		query_id, prob, score, succ = i, query_result_info[i, 0], query_result_info[i, 1], query_result_info[i, 2]
		if prob < VPR_MATCH_THRESHOLD:
			ax[0].bar(query_id, prob, width=0.6, alpha=0.7, label='VPR Score', color='g')
		else:
			ax[0].bar(query_id, prob, width=0.6, alpha=0.7, label='VPR Score', color='r')
		if score > REFINE_EDGE_SCORE_THRESHOLD:
			ax[1].bar(query_id, score, width=0.6, alpha=0.7, label='Edge Score/Loss', color='g')
		else:
			ax[1].bar(query_id, score, width=0.6, alpha=0.7, label='Edge Score/Loss', color='r')
	ax[0].grid(ls='--', color='0.7')
	ax[0].set_title('VPR Score/Loss')
	ax[1].grid(ls='--', color='0.7')
	ax[1].set_title('Edge Score (Green: High Score. Red: Low Score)')
	fig.tight_layout()
	plt.savefig(os.path.join(log_dir, f"preds/results_{query_submap_id}_query_result.png"))

def parse_arguments():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--input_submap_path", type=str, default=None, nargs="+", help="Path to input submaps")
	parser.add_argument("--output_map_path", type=str, default=None, help="Path to output final map")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
										help="Resizing shape for images (WxH). If a single int is passed, set the"
											 "smallest edge of all images to this value, while keeping aspect ratio")
	
	parser.add_argument("--pose_estimation_method", type=str, default="master", 
													help="master, duster")
	parser.add_argument("--vpr_match_model", type=str, default="sequence_match", 
											 help="single_match, topo_filter, sequence_match")

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
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="cuda (gpu) or cpu")
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

