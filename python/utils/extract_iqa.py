#! /usr/bin/env python

"""
extract_iqa.py

This script is designed to work with the map-free dataset format. 
It extracts Image Quality Assessment (IQA) scores for images in the dataset using a specified metric. 
The script supports various IQA metrics provided by the `pyiqa` library and can run on either CPU or GPU.

Usage:
    python extract_iqa.py --map_path path/out_map \
                          --metric musiq --device cuda \
                          --output path/out_map

Arguments:
    --map_path: Path to the dataset directory containing images and poses.
    --metric: The IQA metric to use (default: 'musiq').
    --device: The device to run the metric on (default: 'cuda').
    --output: Directory to save the output IQA scores (default: './results').

The script reads the poses from 'poses.txt' in the dataset directory, computes the IQA score for each image, 
and saves the results in 'iqa.txt' within the specified output directory.
"""

import os
import argparse
import numpy as np
from pathlib import Path
import pyiqa

if __package__:
    from .utils_geom import read_poses
else:
    from utils_geom import read_poses

def main(args):
	seq_path = args.map_path
	if args.output is None:
		out_dir = seq_path
	else:
		out_dir = args.output

	os.makedirs(out_dir, exist_ok=True)

	iqa_metric = pyiqa.create_metric(args.metric, device=args.device)	

	poses = read_poses(os.path.join(seq_path, "poses.txt"))
	scores = []
	
	for indice, (img_name, _) in enumerate(poses.items()):
		img_path = Path(os.path.join(seq_path, img_name))
		if not img_path.exists():
			raise FileNotFoundError(f"Missing {img_path}")
			
		score = iqa_metric(str(img_path)).detach().squeeze().cpu().numpy()
		score_float = float(score)
		scores.append([img_name, score_float])
	
	print(f"Saving IQA scores to {out_dir}")
	scores_array = np.array(scores, dtype=object)
	np.savetxt(os.path.join(out_dir, "iqa_data.txt"), scores_array, fmt="%s %.5f")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Calculate Image Quality Assessment scores")
	parser.add_argument("--map_path", required=True)
	parser.add_argument("--metric", default="musiq", choices=pyiqa.list_models())
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--output", default=None)

	args = parser.parse_args()
	
	main(args)