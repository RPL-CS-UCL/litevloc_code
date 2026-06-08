# [Computation]:
# python submission.py --config ../config/dataset/matterport3d.yaml --split test --out_dir xx --models master --debug
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetricmean/submission.zip --dataset_path /Titan/dataset/data_mapfree --split val --log error

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import time
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back, Style
from transforms3d.quaternions import mat2quat
from estimator import available_models, get_estimator
from estimator.models.base_estimator import BaseEstimator
from estimator.utils import to_numpy

from rpe_default import cfg
from datamodules import DataModule

@dataclass
class PoseResult:
	top_k: int
	reference_image_names: list
	query_image_name: str
	q: np.ndarray
	t: np.ndarray
	conf: float

	def __str__(self) -> str:
		formatter = {"float": lambda v: f"{v:.6f}"}
		max_line_width = 1000
		q_str = np.array2string(
			self.q, formatter=formatter, max_line_width=max_line_width
		)[1:-1]
		t_str = np.array2string(
			self.t, formatter=formatter, max_line_width=max_line_width
		)[1:-1]
		ref_img_names = " ".join(ref_name for ref_name in self.reference_image_names)
		return f"{self.top_k} {ref_img_names} {self.query_image_name} {q_str} {t_str} {self.conf:.3f}"

def predict(loader, estimator, str_estimator, cfg, args):
	results_dict = defaultdict(list)
	results_debug_dict = defaultdict(list)
	running_time = []
	save_indice = 0
	estimator.verbose = False

	for data_cnt, data in enumerate(tqdm(loader)):
		try:
			scene_root = Path(data['scene_root'][0])
			scene_id = data['scene_id'][0]
			if scene_id not in results_dict:
				results_dict[scene_id] = []

			reference_image_names = [name[0] for name in data['list_image0_path']]
			reference_image_poses = [pose.squeeze(0) for pose in data['list_image0_pose']]
			reference_image_intrinsics = [
				{'K': K.squeeze(0), 'im_size': im_size.squeeze(0)} 
				for K, im_size in zip(data['list_K_color0'], data['list_im_size0'])
			]
			
			query_image_name = data['image1_path'][0]
			query_image_intrinsic = {'K': data['K_color1'].squeeze(0), 'im_size': data['im_size1'].squeeze(0)} # K, WxH

			print(Fore.GREEN + f'Scene Root: {scene_root}' + Style.RESET_ALL)
			print(Fore.GREEN + f'Loading Reference Images:', ', '.join(reference_image_names) + Style.RESET_ALL)
			print(Fore.GREEN + f'Loading Query Image: {query_image_name}' + Style.RESET_ALL)

			"""Absolute Pose Estimation"""
			# Images and intrinsics are resized inside the estimator
			est_opts = {
				'known_extrinsics': True,
				'known_intrinsics': False, # False for Joint optimization of intrinsics is better
				'niter': 300,
				'two_stage_opt_niter': 50,
				'crop_image_to_database': args.crop_image_to_database,
				'resize': (512, 288)
			}

			start_time = time.time()
			result = estimator(
				scene_root,
				reference_image_names, query_image_name,
				reference_image_poses, 
				reference_image_intrinsics, query_image_intrinsic,
				est_opts
			)
			estimation_time = time.time() - start_time
			running_time.append(estimation_time)

			"""Definition of solver output"""
			# Rwc (numpy.ndarray): Estimated rotation matrix from world (reference frame) to camera
			# twc (numpy.ndarray): Estimated translation vector. Shape: [3, 1] that translate depth_img1 to depth_img0.
			im_pose, loss_value = result["im_pose"], result["loss"]
			if im_pose is None: 
				raise ValueError(f"{str_estimator} - Estimated pose is None.")
			elif np.isnan(im_pose).any():
				raise ValueError("Estimated pose is NaN or infinite.")
						
			"""Save Results"""
			# Pose that transforms camera point into world
			T_w2c = np.eye(4); T_w2c[:3, :3] = im_pose[:3, :3]; T_w2c[:3, 3] = im_pose[:3, 3]
			T_c2w = np.linalg.inv(T_w2c); rot_c2w = T_c2w[:3, :3]; trans_c2w = T_c2w[:3,  3].reshape(3, 1)

			# populate results_dict
			top_k_matches = len(reference_image_names)
			if hasattr(estimator, 'get_minimum_spanning_tree'):
				msp_edges = estimator.get_minimum_spanning_tree()
				weight_i, weight_j = estimator.scene.weight_i, estimator.scene.weight_j
				for edge in msp_edges:
					if edge[0] == top_k_matches or edge[1] == top_k_matches: # confidence of the query image
						edge_str = f"{edge[0]}_{edge[1]}"
						conf = weight_i[edge_str].mean() * weight_j[edge_str].mean()
			else:
				conf = 0.0

			estimated_pose = PoseResult(
				top_k=top_k_matches,
				reference_image_names=reference_image_names, 
				query_image_name=query_image_name,
				q=mat2quat(rot_c2w).reshape(-1),
				t=trans_c2w.reshape(-1),
				conf=conf
			)
			results_dict[scene_id].append(estimated_pose)

			print(Fore.GREEN + f'({str_estimator}) Estimated Pose in the world: {T_w2c[:3, 3].T}' + Style.RESET_ALL)
			if args.debug:
				output_estimator_directory = Path(os.path.join(args.out_dir, f"{str_estimator}"))
				output_estimator_directory.mkdir(parents=True, exist_ok=True)
				Path(output_estimator_directory / "preds").mkdir(parents=True, exist_ok=True)
				estimator.save_results(output_estimator_directory)
				save_indice += 1
		
			if args.viz:
				estimator.show_reconstruction()
		
		except Exception as e:
			scene = data['scene_id'][0]
			query_image_name = data['image1_path'][0]
			tqdm.write(Fore.RED + f"Error with {str_estimator}: {e}" + Style.RESET_ALL)
			tqdm.write(Fore.RED + f"May occur due to no overlapping regions or insufficient matching at {scene}/{query_image_name}." + Style.RESET_ALL)
			
	average_runtime = running_time[0] if len(running_time) == 1 else np.mean(running_time)
	return results_dict, results_debug_dict, average_runtime

def save_submission(results_dict: dict, output_path: Path):
	with ZipFile(output_path, "w") as zip:
		for scene, poses in results_dict.items():
			poses_str = "#N img0_name1 img0_name2 ... img0_nameN query_image qw qx qy qz tx ty tz confidence\n"
			poses_str += "\n".join((str(pose) for pose in poses))
			zip.writestr(f"pose_{scene}.txt", poses_str.encode("utf-8"))

def eval(args):
	# Load configs
	cfg.merge_from_file(args.config)

	# Create dataloader for different datasets
	if args.split == 'test':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		cfg.DATASET.TOP_K = args.top_k
		cfg.DATASET.N_QUERY = args.n_query
		dataloader = DataModule(cfg).test_dataloader()
	elif args.split == 'val':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		cfg.DATASET.TOP_K = args.top_k
		cfg.DATASET.N_QUERY = args.n_query        
		dataloader = DataModule(cfg).val_dataloader()
	elif args.split == 'train':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		cfg.DATASET.TOP_K = args.top_k
		cfg.DATASET.N_QUERY = args.n_query        
		dataloader = DataModule(cfg).train_dataloader()
	else:
		raise NotImplemented(f'Invalid split: {args.split}')

	output_root = Path(args.out_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	for model in args.models:
		estimator = get_estimator(
			model, 
			device=args.device, 
			out_dir=os.path.join(args.out_dir, f'{model}/preds'),
		)
		results_dict, results_debug_dict, avg_runtime = predict(dataloader, estimator, model, cfg, args)

		if args.debug:
			for scene, values in results_debug_dict.items():
				np.savetxt(os.path.join(args.out_dir, f'{model}/preds', f'debug_{scene}.txt'), np.array(values), fmt='%.5f %.5f')

		print(Fore.GREEN + f"Running APE Method: {model}" + Style.RESET_ALL)

		log_dir = Path(output_root / f"{model}")
		log_dir.mkdir(parents=True, exist_ok=True)

		# Save runtimes to txt
		runtime_str = f"{model}: {avg_runtime:.3f}s"
		with open(log_dir / f"runtime_results_{args.top_k}.txt", "w") as f:
			f.write(runtime_str + "\n")
		tqdm.write(runtime_str)

		# Save predictions to txt per scene within zip
		save_submission(results_dict, log_dir / f"submission_{args.top_k}.zip")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", help="path to config file")
	parser.add_argument(
		"--models",
		type=str,
		nargs="+",
		default="all",
		help=f"Available models: {str(available_models)}"
	)
	parser.add_argument(
		"--device", type=str, default="cuda", choices=["cpu", "cuda"]
	)
	parser.add_argument(
		"--viz",
		action="store_true",
		help="pass --viz to avoid saving visualizations",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="pass --debug to visualize intermediate results",
	)    
	parser.add_argument(
		"--out_dir", type=str, default=None, help="path where outputs are saved"
	)
	parser.add_argument(
		"--num_iters",
		type=int,
		default=1,
		help="number of interations to run benchmark and average over",
	)    
	parser.add_argument(
		"--split",
		choices=("train", "val", "test"),
		default="test",
		help="Dataset split to use for evaluation. Choose from test or val. Default: test",
	)
	parser.add_argument(
		'--top_k', 
		type=int, 
		default=2, 
		help='Number of randomly selected reference images for localization'
	)
	parser.add_argument(
		'--n_query', 
		type=int, 
		default=1, 
		help='Number of query images for localization'
	)
	parser.add_argument(
		'--crop_image_to_database',
		action='store_true',
		help="crop query image to the same size as the database images (especially for cross-device images)",
	)
	args = parser.parse_args()
	if args.models == "all":
		args.models = available_models
	eval(args)
