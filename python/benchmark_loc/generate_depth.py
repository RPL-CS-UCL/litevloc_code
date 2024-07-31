# [Computation]:
# python submission.py --config ../config/dataset/mapfree.yaml --models master --pose_solver essentialmatrixmetric --out_dir /Titan/dataset/data_mapfree/results --split val
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetric/submission.zip --split val --log error

import os
import sys
import argparse
from pathlib import Path

import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map-free-reloc"))
from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.utils.data import data_to_model_device

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ZoeDepth"))

def predict(loader, depth_model, str_depth_model, out_dir):
	running_time = []
	for data in tqdm(loader):
		try:
			str_scene_id = data['scene_id'][0]
			str_img0 = data['pair_names'][0][0]
			str_img1 = data['pair_names'][1][0]
			Path(out_dir).mkdir(parents=True, exist_ok=True)
			Path(out_dir+'/'+str_scene_id).mkdir(parents=True, exist_ok=True)
			Path(out_dir+'/'+str_scene_id+'/seq0').mkdir(parents=True, exist_ok=True)
			Path(out_dir+'/'+str_scene_id+'/seq1').mkdir(parents=True, exist_ok=True)

			if torch.is_tensor(data['image0']): rgb_img0 = data['image0'].to(args.device)
			if torch.is_tensor(data['image1']): rgb_img1 = data['image1'].to(args.device)

			start_time = time.time()
			depth_img0, depth_img1 = depth_model.infer(rgb_img0), depth_model.infer(rgb_img1)
			depth_time = (time.time() - start_time) / 2

			depth_img0_numpy = depth_img0.squeeze().detach().cpu().numpy()
			path_depth0 = os.path.join(out_dir, str_scene_id, str_img0.replace('jpg', 'zoe.png'))
			Image.fromarray((depth_img0_numpy * 1000).astype(np.uint16)).save(path_depth0)

			depth_img1_numpy = depth_img1.squeeze().detach().cpu().numpy()
			path_depth1 = os.path.join(out_dir, str_scene_id, str_img1.replace('jpg', 'zoe.png'))
			Image.fromarray((depth_img1_numpy * 1000).astype(np.uint16)).save(path_depth1)

			running_time.append(depth_time)
		except Exception as e:
			tqdm.write(f"Error with {str_depth_model}: {e}")    
	return np.mean(running_time)

def eval(args):
	"""Load configs"""
	cfg.merge_from_file(args.config)

	"""Data Loader"""
	if args.split == 'test':
		dataloader = DataModule(cfg).test_dataloader()
	elif args.split == 'val':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		dataloader = DataModule(cfg).val_dataloader()
	else:
		raise NotImplemented(f'Invalid split: {args.split}')
	
	"""Depth Model"""
	if args.model == "zoedepth":
		model = torch.hub.load("../../../ZoeDepth", "ZoeD_N", source="local", pretrained=True).to(args.device)
	else:
		return

	print(f"Running Metric Depth Prediction Model: {args.model}")
	avg_runtime = predict(dataloader, model, args.model, args.out_dir)
	print(f"Average runtime: {avg_runtime:.3f}s")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", help="path to config file")
	parser.add_argument(
		"--model", type=str, default="zoedepth", help='zoedepth'
	)
	parser.add_argument(
		"--device", type=str, default="cuda", choices=["cpu", "cuda"]
	)
	parser.add_argument(
		"--out_dir", type=str, default=None, help="path where outputs are saved"
	)
	parser.add_argument(
		"--split",
		choices=("val", "test"),
		default="test",
		help="Dataset split to use for evaluation. Choose from test or val. Default: test",
	)
	args = parser.parse_args()
	eval(args)
