#! /usr/bin/env python

"""
Usage:
python extract_vpr_descriptors.py --dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
--method cosplace --backbone ResNet18 --descriptors_dimension 256 \
--num_preds_to_save 3 \
--image_size 512 288 \
--device cuda \
--save_descriptors

Usage for Jetson: 
python extract_vpr_descriptors.py --dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
--method cosplace --backbone ResNet18 --descriptors_dimension 256 \
--num_preds_to_save 3 \
--image_size 512 288 \
--device cuda \
--save_descriptors
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import time
import matplotlib
from pathlib import Path
import numpy as np
import torch

from image_graph import ImageGraphLoader
from utils.utils_vpr_method import *

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def main(args):
	"""Main function to run the image matching process."""
	for scene in sorted(os.listdir(args.dataset_path)):
		scene_path = os.path.join(args.dataset_path, scene)
		out_dir = scene_path

		# out_dir.mkdir(exist_ok=True, parents=True)
		# log_dir = setup_log_environment(out_dir, args)

		"""Initialize VPR model"""
		model = initialize_vpr_model(args.method, args.backbone, args.descriptors_dimension, args.device)

		"""Load images"""
		image_graph = ImageGraphLoader.load_data(
			Path(scene_path), 
			args.image_size, 
			depth_scale=0.001, 
			load_rgb=True, 
			load_depth=False, 
			normalized=True
		)

		# Extract VPR descriptors for all nodes in the map
		start_time = time.time()
		db_descriptors_id = image_graph.get_all_id()
		db_descriptors = np.empty((image_graph.get_num_node(), args.descriptors_dimension + 1), dtype=object)
		for indices, (_, map_node) in enumerate(image_graph.nodes.items()):
			with torch.no_grad():
				desc = model(map_node.rgb_image.unsqueeze(0).to(args.device)).cpu().numpy()
				vec = np.empty((1, args.descriptors_dimension + 1), dtype=object)
				vec[0, 0], vec[0, 1:] = map_node.rgb_img_name, desc[0]
				db_descriptors[indices, :] = vec[0, :]
		print(f"IDs: {db_descriptors_id} extracted {db_descriptors.shape} VPR descriptors.")
		print(f'Extract each VPR descriptor costs: {(time.time() - start_time) / len(db_descriptors):.3f}s')

		"""Save image descriptors"""
		if args.save_descriptors:
			print(f"Saving image descriptors to {out_dir}")
			np.savetxt(os.path.join(out_dir, f'database_descriptors.txt'), db_descriptors, fmt='%s ' + '%.9f ' * args.descriptors_dimension)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)
