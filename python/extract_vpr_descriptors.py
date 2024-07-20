'''
Usage: python extract_vpr_descriptors.py --dataset_path /Titan/dataset/data_topo_loc/anymal_ops_mos \
--method cosplace --backbone ResNet18 --descriptors_dimension 512 \
--num_preds_to_save 3 \
--image_size 288 512 \
--device cuda \
--save_descriptors
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VPR-methods-evaluation'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VPR-methods-evaluation/third_party/deep-image-retrieval'))

import time
import argparse
import matplotlib
from pathlib import Path
import numpy as np
import torch

from utils.utils_vpr_method import *
from utils.utils_image import load_rgb_image
from image_graph import ImageGraphLoader
from image_node import ImageNode

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def main(args):
	"""Main function to run the image matching process."""
	out_dir = Path(os.path.join(args.dataset_path, 'output_extract_vpr_descriptors'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	"""Initialize VPR model"""
	model = initialize_vpr_model(args.method, args.backbone, args.descriptors_dimension, args.device)

	"""Load images"""
	image_graph = ImageGraphLoader.load_data(os.path.join(args.dataset_path, 'map'), 
																					 image_size=args.image_size, 
																					 depth_scale=0.001,
																					 normalized=True)

	# Extract VPR descriptors for all nodes in the map
	db_descriptors_id = image_graph.get_all_id()
	db_descriptors = np.empty((image_graph.get_num_node(), args.descriptors_dimension), dtype="float32")
	for indices, (_, map_node) in enumerate(image_graph.nodes.items()):
		with torch.no_grad():
			desc = model(map_node.rgb_image.unsqueeze(0).to(args.device)).cpu().numpy()
			db_descriptors[indices, :] = desc[0]
	print(f"IDs: {db_descriptors_id} extracted {db_descriptors.shape} VPR descriptors.")

	"""Save image descriptors"""
	if args.save_descriptors:
		np.save(os.path.join(log_dir, 'preds', "database_descriptors.npy"), db_descriptors)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)
