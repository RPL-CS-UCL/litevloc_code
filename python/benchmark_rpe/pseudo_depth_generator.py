import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation

from datamodules import DataModule
from estimator import get_estimator
from rpe_default import cfg  # Import configuration settings

def process_data(loader, estimator, args):
	"""Process data batches to estimate and save depth maps."""
	existing_pairs = dict() # existing_pairs[scene_name] store the pair name
	finetune_split = dict() # finetune_split[scene_name] store the list of finetune name of a scene
	data_pairs = dict()     # data_pairs[scene_name] store the list of finetune pairs of a scene
	est_poses = dict()

	Path(os.path.join(args.out_dir, 'pairs')).mkdir(parents=True, exist_ok=True)
	Path(os.path.join(args.out_dir, 'split')).mkdir(parents=True, exist_ok=True)
	for data in tqdm(loader):
		try:
			scene_root = Path(data['scene_root'][0])
			if scene_root.name not in existing_pairs:
				existing_pairs[scene_root.name] = set()
				finetune_split[scene_root.name] = set()
				data_pairs[scene_root.name] = list()
				est_poses[scene_root.name] = list()

			list_img_name = [name[0] for name in data['list_image0_path']]
			list_img_poses = [pose.squeeze(0) for pose in data['list_image0_pose']]
			list_img_intr = [{'K': K.squeeze(0), 'im_size': im_size.squeeze(0)} \
							  for K, im_size in zip(data['list_K_color0'], data['list_im_size0'])]
			print(f"Generate depth maps using {list_img_name}")
			
			# Run depth estimation
			est_opts = {'known_extrinsics': True, 'known_intrinsics': False, 'resize': 512}
			estimator.niter = 300
			estimator(
				scene_root,
				list_img_name, None,
				list_img_poses, list_img_intr,
				None,
				est_opts
			)
			# estimator.show_reconstruction()
		except Exception as e:
			print(f"Error processing: {e}")
			continue

		# weight_i['i_j']: the calibrated confidence of ith image with i_j pair
		# weight_j['i_j']: the calibrated confidence of jth image with i_j pair
		# weight map for pair (0, 1) -> weight_i['0_1'], weight_j['0_1']
		conf_i, conf_j = estimator.scene.conf_i, estimator.scene.conf_j
		weight_i, weight_j = estimator.scene.weight_i, estimator.scene.weight_j

		# Color
		# colors = [(c.clip(min=0, max=1) * 255).astype(np.uint8) for c in estimator.scene.imgs]

		# Retrieve and process depth map and estimated poses
		depth_maps = estimator.scene.get_depthmaps()
		depths = [(d.detach().cpu().numpy() * 1000.0).astype(np.uint16) for d in depth_maps]
		list_depth_name = [name.replace('.jpg', '.pdepth.png') for name in list_img_name]

		im_poses = [(pose.detach().cpu().numpy()) for pose in estimator.scene.get_im_poses()] # world -> camera TF
		pose_init = None
		for img_name, pose_ct2w in zip(list_img_name, im_poses):
			if pose_init is None:
				pose_init = copy.deepcopy(pose_ct2w)
			pose_c02ct = np.linalg.inv(pose_ct2w) @ pose_init
			trans = pose_c02ct[:3, 3]
			quat = np.roll(Rotation.from_matrix(pose_c02ct[:3, :3]).as_quat(), 1)
			est_poses[scene_root.name].append([img_name] + quat.tolist() + trans.tolist())

		# The connectivity graph for computation
		msp_edges = estimator.get_minimum_spanning_tree()
		print('MST edges for Pose Estimation: ', msp_edges)

		edge_scores = estimator.get_similarity()
		sorted_edge_scores = dict(sorted(edge_scores.items(), key=lambda item: item[1], reverse=True))
		max_score = next(iter(sorted_edge_scores.values()))

		for edge_str, score in sorted_edge_scores.items():
			if score < max_score * 0.3:
				continue
			
			print(f'Edges: {edge_str}')

			edge = [int(edge_str.split('_')[0]), int(edge_str.split('_')[1])]
			confs = [conf_i[edge_str].detach().cpu().numpy(), conf_j[edge_str].detach().cpu().numpy()]
			weights = [weight_i[edge_str].detach().cpu().numpy(), weight_j[edge_str].detach().cpu().numpy()]
			valid_masks = [w >= estimator.calib_params['pseudo_gt_thre'] for w in weights]

			SIZE_THRE = 0.2 # reliable match threshold - outdoor setting: 0.3; indoor setting: 0.65
			for m, d in zip(valid_masks, depths):
				print(f"{np.sum(m):.3f}, {d.size}, {d.size * SIZE_THRE:.3f}")

			if all(np.sum(m) >= d.size * SIZE_THRE for m, d in zip(valid_masks, depths)):
				for idx in range(len(edge)):
					if list_depth_name[edge[idx]] not in existing_pairs[scene_root.name]:
						existing_pairs[scene_root.name].add(list_depth_name[edge[idx]])

						# Filter out unreliable depth
						depth = depths[edge[idx]]; depth[~valid_masks[idx]] = 0
						# Resize the depth image to the original size
						new_size = tuple(list_img_intr[edge[idx]]['im_size'].cpu().numpy().astype(int)) # WxH
						re_depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
						
						output_path = Path(args.out_dir) / 'pairs' / scene_root.name / list_depth_name[edge[idx]]
						output_path.parent.mkdir(parents=True, exist_ok=True)

						cv2.imwrite(str(output_path), re_depth)
						print(f'Saving pdepth to {str(output_path)}')

						# Plot and save confidence map
						output_path = Path(args.out_dir) / 'confs' / scene_root.name / list_depth_name[edge[idx]].replace('.pdepth.png', '.raw_conf.jpg')
						output_path.parent.mkdir(parents=True, exist_ok=True)
						conf_map = confs[idx]
						fig, ax = plt.subplots(1, 1, figsize=(5, 5))
						im = ax.imshow(conf_map, cmap='jet')
						ax.set_title('Confidence Map')
						fig.colorbar(im, ax=ax)
						plt.savefig(str(output_path))
						plt.close(fig)  # Close the figure to free memory

						# Plot and save weight map
						output_path = Path(args.out_dir) / 'confs' / scene_root.name / list_depth_name[edge[idx]].replace('.pdepth.png', '.calib_conf.jpg')
						output_path.parent.mkdir(parents=True, exist_ok=True)
						weight_map = weights[idx]
						fig, ax = plt.subplots(1, 1, figsize=(5, 5))
						im = ax.imshow(weight_map, cmap='jet')
						ax.set_title('Weight Map')
						fig.colorbar(im, ax=ax)
						plt.savefig(str(output_path))
						plt.close(fig)  # Close the figure to free memory

				finetune_split[scene_root.name].add(list_img_name[edge[0]])
				finetune_split[scene_root.name].add(list_img_name[edge[1]])

				data_pairs[scene_root.name].append((
					scene_root.name, 
					list_img_name[edge[0]], 
					list_img_name[edge[1]], 
					list_depth_name[edge[0]], 
					list_depth_name[edge[1]])
				)

	for scene_name, split in finetune_split.items():
		print(f"The number of finetun split for scene {scene_name}: {len(split)}")
		output_path = Path(args.out_dir, 'split', f"finetune_{scene_name}_split.txt")
		output_path.parent.mkdir(parents=True, exist_ok=True)
		np.savetxt(str(output_path), np.array(list(split)).reshape(-1, 1), fmt="%s")

	for scene_name, poses in est_poses.items():
		print(f"The number of est_poses for scene {scene_name}: {len(poses)}")
		output_path = Path(args.out_dir, 'pairs', f"{scene_name}", 'poses_pseudo.txt')
		output_path.parent.mkdir(parents=True, exist_ok=True)
		np.savetxt(str(output_path), np.array(poses, dtype=object), fmt="%.3f " * 7)

	dtype = [('scene_name', 'U20'), ('img0', 'U50'), ('img1', 'U50'), ('depth0', 'U50'), ('depth1', 'U50')]
	for scene_name, pairs in data_pairs.items():
		print(f"Total pairs for scene {scene_name} are generated: {len(pairs)}")
		output_path = Path(os.path.join(args.out_dir, 'pairs', f"mapfree_pairs_{scene_name}_pdepth.npy"))
		output_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(str(output_path), np.array(pairs, dtype=dtype))

		if args.save_gtpair:
			pairs_gtdepth = [(
				pair[0], 
				pair[1], 
				pair[2],
				pair[3].replace('.pdepth.png', f'.{cfg.DATASET.ESTIMATED_DEPTH}.png'),
				pair[4].replace('.pdepth.png', f'.{cfg.DATASET.ESTIMATED_DEPTH}.png')) for pair in pairs
			]
			np.save(str(output_path).replace('pdepth', 'gtdepth'), np.array(pairs_gtdepth, dtype=dtype))

			pairs_m3ddepth = [(
				pair[0], 
				pair[1], 
				pair[2],
				pair[3].replace('.pdepth.png', f'.metric3d.png'),
				pair[4].replace('.pdepth.png', f'.metric3d.png')) for pair in pairs
			]
			np.save(str(output_path).replace('pdepth', 'm3ddepth'), np.array(pairs_m3ddepth, dtype=dtype))

def main(args):
	"""Main pipeline setup and execution."""
	# Load configuration file
	cfg.merge_from_file(args.config)

	# Configure and initialize data loader
	cfg.TRAINING.BATCH_SIZE = 1
	cfg.TRAINING.NUM_WORKERS = 1
	cfg.DATASET.TOP_K = args.top_k
	cfg.DATASET.N_QUERY = args.n_query
	dataloader = DataModule(cfg).test_dataloader()

	# Initialize depth estimation model
	estimator = get_estimator(args.model, device=args.device)
	estimator.verbose = True
	estimator.set_calib_params(dict(mu=1.0, conf_thre=0.5, pseudo_gt_thre=args.pseudo_gt_thre, use_weight_opt=True))
	assert (estimator.calib_params is not None), "Should use duster_calib_pretrain or master_calib_pretrain"

	# Process all images
	process_data(dataloader, estimator, args)

if __name__ == "__main__":
	# Configure command-line arguments
	parser = argparse.ArgumentParser(description="Pseudo Depth Map Generator")
	parser.add_argument("--config", required=True, help="Path to config.yaml")
	parser.add_argument("--model", required=True, help="Model selection")
	parser.add_argument("--out_dir", required=True, help="Output directory")
	parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
	parser.add_argument("--top_k", type=int, default=2, help="Number of reference images")
	parser.add_argument("--n_query", type=int, default=1, help="Number of query images")
	parser.add_argument("--save_gtpair", action="store_true")
	parser.add_argument("--pseudo_gt_thre", type=float, default=1.5, help="Pseudo gt threshold for specific datasets")
	
	args = parser.parse_args()
	main(args)