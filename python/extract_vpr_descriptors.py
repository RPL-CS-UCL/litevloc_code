'''
Usage: python test_batch_vpr_method.py --dataset_path /Titan/dataset/data_topo_loc/anymal_ops_mos \
--method cosplace --backbone ResNet18 --descriptors_dimension 512 \
--num_preds_to_save 3 \
--image_size 288 512 \
--device cuda \
--sample_map 3 --sample_obs 1000
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
	out_dir = Path(os.path.join(args.dataset_path, 'output_batch_vpr_method'))
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
	for map_id, map_node in image_graph.
	db_descriptors_id = image_graph.get_all_id()
	db_descriptors = np.array([map_node.get_descriptor() for _, map_node in image_graph.nodes.items()], dtype="float32")
	print(f"IDs: {db_descriptors_id} extracted {db_descriptors.shape} VPR descriptors.")

	"""Main loop for processing observations"""
	obs_poses_gt = np.loadtxt(os.path.join(args.dataset_path, 'obs', 'camera_pose_gt.txt'))
	for obs_id in range(0, len(obs_poses_gt), 5):
		if obs_id > 10:
			break

		# Load observation data
		print(f"obs_id: {obs_id}")
		rgb_img_path = os.path.join(args.dataset_path, 'obs/rgb', f'{obs_id:06d}.png')
		rgb_img = load_rgb_image(rgb_img_path, args.image_size, normalized=False)
		depth_img_path = os.path.join(args.dataset_path, 'obs/depth', f'{obs_id:06d}.png')
		depth_img = load_depth_image(depth_img_path, args.image_size, depth_scale=args.depth_scale)
		with torch.no_grad():
			desc = vpr_model(rgb_img.unsqueeze(0).to(args.device)).cpu().numpy()
		obs_node = ImageNode(obs_id, rgb_img, depth_img, desc, 0,
						 						 np.zeros(3), np.array([0, 0, 0, 1]),
						 						 rgb_img_path, depth_img_path)
		obs_node.set_pose_gt(obs_poses_gt[obs_id, 1:4], obs_poses_gt[obs_id, 4:])

		"""Extract descriptors"""
		start_time = time.time()
		queries_descriptors = np.empty((1, args.vpr_descriptors_dimension), dtype="float32")
		queries_descriptors[0] = query_desc
		_, predictions = perform_knn_search(
			db_descriptors,
			queries_descriptors,
			args.vpr_descriptors_dimension,
			args.recall_values
		)	
		print('Matching desc costs: {:3f}s'.format((time.time() - start_time)))

		"""Save image descriptors"""
		if args.save_descriptors:
			save_descriptors(log_dir, queries_descriptors, database_descriptors)

		"""Save visualizations of predictions."""
		if args.num_preds_to_save != 0:
			logging.info("Saving final predictions")
			list_of_images_paths = [obs_node.rgb_img_path]
			for j in range(len(predictions[0][:args.num_preds_to_save])):
				if predictions[0][j] < 0: continue
				map_node = image_graph.get_node(all_map_id[predictions[0][j]])
				if map_node is not None:
					list_of_images_paths.append(map_node.rgb_img_path)

			preds_correct = [None] * len(list_of_images_paths)
			save_visualization(log_dir, obs_id, list_of_images_paths, preds_correct)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)
