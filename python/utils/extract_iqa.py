#! /usr/bin/env python

"""
Usage: work with map-free dataset format
python extract_iqa.py --dataset_path path/out_map \
--metric musiq --device cuda \
--output path/out_map
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import argparse
import numpy as np
from pathlib import Path
import pyiqa
from utils.utils_geom import read_poses

def main(args):
	iqa_metric = pyiqa.create_metric(args.metric, device=args.device)
	
	dataset_path = Path(args.dataset_path)
	poses = read_poses(dataset_path/"poses.txt")
	scores = np.empty((len(poses), 2), dtype=object)
	for indice, (img_name, _) in enumerate(poses.items()):
		img_path = dataset_path / img_name
		if not img_path.exists():
			raise FileNotFoundError(f"Missing {img_path}")
			
		score = iqa_metric(str(img_path)).detach().squeeze().cpu().numpy()
		scores[indice, 0], scores[indice, 1] = img_name, score
	
	out_path = Path(args.output)
	out_path.mkdir(parents=True, exist_ok=True)
	np.savetxt(out_path/"iqa.txt", scores, fmt="%s %.5f")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", required=True)
	parser.add_argument("--metric", default="musiq", choices=pyiqa.list_models())
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--output", default="./results")
	args = parser.parse_args()
	
	main(args)