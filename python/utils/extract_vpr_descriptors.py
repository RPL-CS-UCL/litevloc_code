#! /usr/bin/env python

"""
Usage:
python extract_vpr_descriptors.py --map_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
--method cosplace --backbone ResNet18 --descriptors_dimension 256 \
--image_size 512 288 --device cuda --num_preds_to_save 3

Usage for Jetson: 
python extract_vpr_descriptors.py --map_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
--method cosplace --backbone ResNet18 --descriptors_dimension 256 \
--num_preds_to_save 3 \
--image_size 512 288 \
--device cuda
"""

import os
import sys
import time
import matplotlib
from pathlib import Path
import numpy as np
import torch

if __package__:
    from .utils_vpr_method import parse_arguments, initialize_vpr_model
    from .image_graph import ImageGraphLoader
else:
    from utils_vpr_method import parse_arguments, initialize_vpr_model
    from image_graph import ImageGraphLoader

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def main(args):
	"""Main function to run the image matching process."""
	seq_path = args.map_path

	"""Initialize VPR model"""
	model = initialize_vpr_model(args.method, args.backbone, args.descriptors_dimension, args.device)

	"""Load images"""
	image_graph = ImageGraphLoader.load_data(
		Path(seq_path), 
		args.image_size, 
		depth_scale=0.001, 
		load_rgb=True, 
		load_depth=False, 
		normalized=True,
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
	print(f"Saving image descriptors to {seq_path}")
	np.savetxt(
		os.path.join(seq_path, f'database_descriptors.txt'), 
		db_descriptors, 
		fmt='%s ' + '%.9f ' * (db_descriptors.shape[1]-1)
	)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)
