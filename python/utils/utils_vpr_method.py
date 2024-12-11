import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation/third_party/deep-image-retrieval'))

from datetime import datetime
import logging
import numpy as np

import vpr_models
from visualizations import build_prediction_image, save_file_with_paths

import argparse
import faiss
from matplotlib import pyplot as plt

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
	tmp_dir = os.path.join(out_dir, f'outputs_{args.method}')
	log_dir = os.path.join(tmp_dir, f'{args.backbone}_' + start_time.strftime('%Y-%m-%d_%H-%M-%S'))
	setup_logging(log_dir, stdout_level="info")
	logging.info(" ".join(sys.argv))
	logging.info(f"Arguments: {args}")
	logging.info(f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}")
	logging.info(f"The outputs are being saved in {log_dir}")
	os.makedirs(os.path.join(log_dir, 'preds'))
	os.system(f"rm {os.path.join(tmp_dir, 'latest')}")
	os.system(f"ln -s {log_dir} {os.path.join(tmp_dir, 'latest')}")
	return log_dir

def initialize_vpr_model(method, backbone, descriptors_dimension, device):
	"""Initialize and return the model."""
	model = vpr_models.get_model(method, backbone, descriptors_dimension)
	return model.eval().to(device)

def perform_knn_search(database_descriptors, queries_descriptors, descriptors_dimension, recall_values):
	"""Perform kNN search and return predictions."""
	faiss_index = faiss.IndexFlatL2(descriptors_dimension)
	faiss_index.add(database_descriptors)
	logging.info("Calculating recalls")
	distances, predictions = faiss_index.search(queries_descriptors, max(recall_values))
	return distances, predictions

def compute_euclidean_dis(query_descriptor, database_descriptor):
	desc1 = query_descriptor.reshape(-1)
	desc2 = database_descriptor.reshape(-1)
	dis = np.linalg.norm(desc1 - desc2)
	return dis

def compute_cosine_similarity(query_descriptor, database_descriptor):
	"""Compute cosine similarity between query and database descriptors."""
	desc1 = query_descriptor.reshape(-1)
	desc2 = database_descriptor.reshape(-1)
	sim = np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
	return sim

def save_descriptors(log_dir, queries_descriptors, database_descriptors):
	"""Save descriptors to files."""
	logging.debug(f"Saving the descriptors in {log_dir}")
	np.save(os.path.join(log_dir, 'preds', "queries_descriptors.npy"), queries_descriptors)
	np.save(os.path.join(log_dir, 'preds', "database_descriptors.npy"), database_descriptors)

def save_visualization(log_dir, query_index, list_of_images_paths, preds_correct):
	"""Save visualization to files."""
	logging.debug(f"Saving the visualization in {log_dir}")
	prediction_image = build_prediction_image(list_of_images_paths, preds_correct)
	pred_image_path = os.path.join(log_dir, 'preds', f"vpr_{query_index:06d}.jpg")
	prediction_image.save(pred_image_path)
	save_file_with_paths(
		query_path=list_of_images_paths[0],
		preds_paths=list_of_images_paths[1:],
		positives_paths=None,
		output_path=os.path.join(log_dir, 'preds', f"vpr_{query_index:06d}.txt"),
		use_labels=False
	)

def save_vis_brief_function(log_dir, belief, niter):
	ids = np.arange(len(belief))
	plt.figure(figsize=(10, 6))
	plt.bar(ids, belief, width=0.6, alpha=0.7, label='Belief Probability')
	plt.xlabel('Frame ID', fontsize=12)
	plt.ylabel('Probability', fontsize=12)
	plt.title(f'Belief Distribution at {niter} Query', fontsize=14)
	plt.xticks(ids, fontsize=10)
	plt.yticks(fontsize=10)
	plt.legend(fontsize=12)
	plt.grid(axis='y', linestyle='--', alpha=0.7)    
	pred_belief_path = os.path.join(log_dir, 'preds', f"belief_{niter}.png")
	plt.savefig(pred_belief_path)

def save_vis_diff_matrix(log_dir, diff_matrix):
	plt.figure(figsize=(8, 8))
	plt.imshow(diff_matrix, cmap='viridis', aspect='auto')
	plt.colorbar(label='Difference')
	plt.xlabel('Database Descriptor Index')
	plt.ylabel('Query Descriptor Index')
	plt.title('Difference Matrix')
	diff_matrix_path = os.path.join(log_dir, 'preds', "diff_matrix.png")
	plt.savefig(diff_matrix_path)

def parse_arguments():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
											help="Resizing shape for images (WxH). If a single int is passed, set the"
											"smallest edge of all images to this value, while keeping aspect ratio")

	parser.add_argument("--positive_dist_threshold", type=int, default=25,
											help="distance (in meters) for a prediction to be considered a positive")
	parser.add_argument("--method", type=str, default="cosplace",
											choices=["netvlad", "apgem", "sfrs", "cosplace", "convap", "mixvpr", "eigenplaces", 
																"eigenplaces-indoor", "anyloc", "salad", "salad-indoor", "cricavpr"],
											help="_")
	parser.add_argument("--backbone", type=str, default=None,
											choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
											help="_")
	parser.add_argument("--descriptors_dimension", type=int, default=None,
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
	args = parser.parse_args()
	
	args.use_labels = not args.no_labels
	
	if args.method == "netvlad":
		if args.backbone not in [None, "VGG16"]:
			raise ValueError("When using NetVLAD the backbone must be None or VGG16")
		if args.descriptors_dimension not in [None, 4096, 32768]:
			raise ValueError("When using NetVLAD the descriptors_dimension must be one of [None, 4096, 32768]")
		if args.descriptors_dimension is None:
			args.descriptors_dimension = 4096
			
	elif args.method == "sfrs":
		if args.backbone not in [None, "VGG16"]:
			raise ValueError("When using SFRS the backbone must be None or VGG16")
		if args.descriptors_dimension not in [None, 4096]:
			raise ValueError("When using SFRS the descriptors_dimension must be one of [None, 4096]")
		if args.descriptors_dimension is None:
			args.descriptors_dimension = 4096
	
	elif args.method == "cosplace":
		if args.backbone is None:
			args.backbone = "ResNet50"
		if args.descriptors_dimension is None:
			args.descriptors_dimension = 512
		if args.backbone == "VGG16" and args.descriptors_dimension not in [64, 128, 256, 512]:
			raise ValueError("When using CosPlace with VGG16 the descriptors_dimension must be in [64, 128, 256, 512]")
		if args.backbone == "ResNet18" and args.descriptors_dimension not in [32, 64, 128, 256, 512]:
			raise ValueError("When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]")
		if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [32, 64, 128, 256, 512, 1024, 2048]:
			raise ValueError(f"When using CosPlace with {args.backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]")
	
	elif args.method == "convap":
		if args.backbone is None:
			args.backbone = "ResNet50"
		if args.descriptors_dimension is None:
			args.descriptors_dimension = 512
		if args.backbone not in [None, "ResNet50"]:
			raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
		if args.descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
			raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]")
	
	elif args.method == "mixvpr":
		if args.backbone is None:
			args.backbone = "ResNet50"
		if args.descriptors_dimension is None:
			args.descriptors_dimension = 512
		if args.backbone not in [None, "ResNet50"]:
			raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
		if args.descriptors_dimension not in [None, 128, 512, 4096]:
			raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 128, 512, 4096]")
	
	elif args.method == "eigenplaces":
		if args.backbone is None:
			args.backbone = "ResNet50"
		if args.descriptors_dimension is None:
			args.descriptors_dimension = 512
		if args.backbone == "VGG16" and args.descriptors_dimension not in [512]:
			raise ValueError("When using EigenPlaces with VGG16 the descriptors_dimension must be in [512]")
		if args.backbone == "ResNet18" and args.descriptors_dimension not in [256, 512]:
			raise ValueError("When using EigenPlaces with ResNet18 the descriptors_dimension must be in [256, 512]")
		if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [128, 256, 512, 2048]:
			raise ValueError(f"When using EigenPlaces with {args.backbone} the descriptors_dimension must be in [128, 256, 512, 2048]")
				
	elif args.method == "eigenplaces-indoor":
		args.backbone = "ResNet50"
		args.descriptors_dimension = 2048
	
	elif args.method == "apgem":
		args.backbone = "Resnet101"
		args.descriptors_dimension = 2048
	
	elif args.method == "anyloc":
		args.backbone = "DINOv2"
		args.descriptors_dimension = 49152
	
	elif args.method == "salad":
		args.backbone = "DINOv2"
		args.descriptors_dimension = 8448
			
	elif args.method == "salad-indoor":
		args.backbone = "Dinov2"
		args.descriptors_dimension = 8448
	
	elif args.method == "cricavpr":
		args.backbone = "Dinov2"
		args.descriptors_dimension = 10752
	
	if args.image_size and len(args.image_size) > 2:
		raise ValueError(f"The --image_size parameter can only take up to 2 values, but has received {len(args.image_size)}.")
	
	return args

