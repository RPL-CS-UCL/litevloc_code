#! /usr/bin/env python

"""Coordinate transformation utilities between g2o, MapFree, and TUM pose formats.

This script provides conversion between three pose representation formats:
- g2o: Graph optimization format used in SLAM
- MapFree: Custom format with image-timestamped poses
- TUM: Technical University of Munich trajectory format
"""

import argparse
import numpy as np
from utils_geom import convert_matrix_to_vec, convert_vec_to_matrix, read_poses, read_timestamps


def convert_g2o_to_mapfree(
	g2o_input_path: str, 
	mapfree_output_path: str
) -> None:
	"""Converts g2o SLAM poses to MapFree format with image-based timestamps.

	Args:
		g2o_input_path: Path to input g2o file
		mapfree_output_path: Path to output MapFree format pose file
	"""
	pose_entries = []
	
	with open(g2o_input_path, 'r') as file:
		for line in file:
			if not line.startswith('VERTEX_SE3:QUAT'):
				continue

			# Parse g2o line: ID, translation (3), quaternion (4)
			components = line.strip().split()
			node_id = int(components[1])
			translation = np.array(list(map(float, components[2:5])))
			quaternion = np.array(list(map(float, components[5:9])))
			
			# Convert to camera-to-world transformation
			world_to_cam = convert_vec_to_matrix(translation, quaternion, 'xyzw')
			cam_to_world = np.linalg.inv(world_to_cam)
			cam_trans, cam_quat = convert_matrix_to_vec(cam_to_world, 'wxyz')
			
			# Create MapFree format entry
			image_name = f'seq/{node_id:06}.color.jpg'
			pose_entry = [image_name, *cam_quat, *cam_trans]
			pose_entries.append(pose_entry)

	# Save in MapFree format with image name as first column
	np.savetxt(mapfree_output_path, pose_entries, fmt='%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f')
	print(f"Converted {g2o_input_path} to MapFree format at {mapfree_output_path}")

def convert_mapfree_to_tum(
	mapfree_pose_path: str, 
	timestamp_path: str, 
	tum_output_path: str
) -> None:
	"""Converts MapFree poses to TUM format with timestamps.

	Args:
		mapfree_pose_path: Path to MapFree pose file
		timestamp_path: Path to timestamp mapping file
		tum_output_path: Output path for TUM-format trajectory
	"""
	pose_dict = read_poses(mapfree_pose_path)
	timestamp_dict = read_timestamps(timestamp_path)
	tum_entries = []

	for img_name in pose_dict:
		if img_name not in timestamp_dict:
			continue

		# Extract camera-to-world pose components
		quat = pose_dict[img_name][0:4]
		trans = pose_dict[img_name][4:7]
		
		# Convert to world-to-camera for TUM format
		cam_to_world = convert_vec_to_matrix(trans, quat, 'wxyz')
		world_to_cam = np.linalg.inv(cam_to_world)
		tum_trans, tum_quat = convert_matrix_to_vec(world_to_cam, 'xyzw')
		
		# Create TUM entry: [timestamp, translation, quaternion]
		tum_entry = [*timestamp_dict[img_name], *tum_trans, *tum_quat]
		tum_entries.append(tum_entry)

	print(np.array(tum_entries).shape)
	np.savetxt(tum_output_path, tum_entries, fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f')


def convert_tum_to_mapfree(
	tum_input_path: str, 
	mapfree_pose_path: str, 
	timestamp_output_path: str
) -> None:
	"""Converts TUM trajectory format to MapFree pose format.

	Args:
		tum_input_path: Path to TUM-format trajectory file
		mapfree_pose_path: Output path for MapFree poses
		timestamp_output_path: Output path for image-timestamp mapping
	"""
	mapfree_poses = []
	timestamps = []

	with open(tum_input_path, 'r') as file:
		for idx, line in enumerate(file):
			data = line.strip().split()
			if len(data) != 8:
				continue

			try:  # Handle timestamp or image name in first column
				timestamp = float(data[0])
				image_name = f"{idx:06d}.jpg"
			except ValueError:
				timestamp = 0.0
				image_name = data[0]

			# Store timestamp mapping
			timestamps.append((image_name, timestamp))

			# Convert TUM pose to camera-to-world
			translation = np.array(list(map(float, data[1:4])))
			quaternion = np.array(list(map(float, data[4:8])))
			world_to_cam = convert_vec_to_matrix(translation, quaternion, 'xyzw')
			cam_to_world = np.linalg.inv(world_to_cam)
			cam_trans, cam_quat = convert_matrix_to_vec(cam_to_world, 'wxyz')

			# Create MapFree entry: [image_name, quat_wxyz, trans_xyz]
			mapfree_poses.append([image_name, *cam_quat, *cam_trans])

	# Save outputs
	timestamp_dtype = [('image_name', 'U20'), ('timestamp', 'f8')]
	np.savetxt(timestamp_output_path, np.array(timestamps, dtype=timestamp_dtype), 
			   fmt='%s %.9f')
	np.savetxt(mapfree_pose_path, mapfree_poses, 
			   fmt='%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f')

def main():
	"""Main entry point for pose format conversion."""
	parser = argparse.ArgumentParser(
		description='Coordinate transformation format converter',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	# Input/output specifications
	parser.add_argument('--input_pose', type=str, required=True,
						help='Path to input pose file')
	parser.add_argument('--output_pose', type=str, required=True,
						help='Path to output pose file')
	parser.add_argument('--input_type', type=str, required=True,
						choices=['g2o', 'mapfree', 'tum'],
						help='Format of input pose file')
	parser.add_argument('--output_type', type=str, required=True,
						choices=['g2o', 'mapfree', 'tum'],
						help='Desired output format')
	
	# Optional arguments for timestamp files
	parser.add_argument('--input_time', type=str,
						help='Path to input timestamp file (MapFree->TUM)')
	parser.add_argument('--output_time', type=str,
						help='Path to output timestamp file (TUM->MapFree)')

	args = parser.parse_args()

	# Route to appropriate conversion function
	if args.input_type == 'g2o' and args.output_type == 'mapfree':
		convert_g2o_to_mapfree(args.input_pose, args.output_pose)
	elif args.input_type == 'mapfree' and args.output_type == 'tum':
		if not args.input_time:
			raise ValueError("MapFree->TUM conversion requires --input_time")
		convert_mapfree_to_tum(args.input_pose, args.input_time, args.output_pose)
	elif args.input_type == 'tum' and args.output_type == 'mapfree':
		if not args.output_time:
			raise ValueError("TUM->MapFree conversion requires --output_time")
		convert_tum_to_mapfree(args.input_pose, args.output_pose, args.output_time)
	else:
		raise NotImplementedError(f"Conversion from {args.input_type} to {args.output_type} not implemented")


if __name__ == '__main__':
	main()
