#! /usr/bin/env python

import os
import numpy as np
from scipy.spatial.transform import Rotation

def read_timestamps(file_path):
	times = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = float(line.strip().split(' ')[1]) # Each row: image_name, timestamp
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				data = float(line.strip().split(' ')[1]) # Each row: qw, qx, qy, tx, ty, tz
			times[img_name] = np.array(data)
	return times

def read_poses(file_path):
	if not os.path.exists(file_path):
		print(f"Poses not found in {file_path}")
		return None

	poses = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, qw, qx, qy, tx, ty, tz
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				data = [float(p) for p in line.strip().split(' ')] # Each row: qw, qx, qy, tx, ty, tz
			poses[img_name] = np.array(data)
	return poses

def read_intrinsics(file_path):
	if not os.path.exists(file_path):
		print(f"Intrinsics not found in {file_path}")
		return None

	intrinsics = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('#'): 
				continue
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, fx fy cx cy width height
			else:
				img_name = f'{line_id:06}.color.jpg'
				data = [float(p) for p in line.strip().split(' ')] # Each row: fx fy cx cy width height
			intrinsics[img_name] = np.array(data)
	return intrinsics

def read_descriptors(file_path):
	if not os.path.exists(file_path):
		print(f"Descriptors not found in {file_path}")
		return None
	
	descs = dict()
	with open(file_path, 'r') as f:
		for line_id, line in enumerate(f):
			if line.startswith('seq'):
				img_name = line.strip().split(' ')[0]
				data = [float(p) for p in line.strip().split(' ')[1:]] # Each row: image_name, descriptor (a vector)
				descs[img_name] = np.array(data)
			else:
				img_name = f'seq/{line_id:06}.color.jpg'
				descs[img_name] = np.array([float(p) for p in line.strip().split(' ')])
	return descs

# Function to convert position and quaternion vectors to a transformation matrix
def convert_vec_to_matrix(vec_p, vec_q, mode='xyzw'):
	# Initialize a 4x4 identity matrix
	tf = np.eye(4)
	if mode == 'xyzw':
		# vec_p: position vector (x, y, z)
		# vec_q: quaternion vector (qx, qy, qz, qw)
		# Set the rotation part of the transformation matrix using the quaternion
		tf[:3, :3] = Rotation.from_quat(vec_q).as_matrix()
		# Set the translation part of the transformation matrix
		tf[:3, 3] = vec_p
	elif mode == 'wxyz':
		# vec_p: position vector (x, y, z)
		# vec_q: quaternion vector (qw, qx, qy, qz)
		# Set the rotation part of the transformation matrix using the quaternion
		tf[:3, :3] = Rotation.from_quat(np.roll(vec_q, -1)).as_matrix()
		# Set the translation part of the transformation matrix
		tf[:3, 3] = vec_p
	return tf

# Function to convert a transformation matrix back to position and quaternion vectors
# tf_matrix: 4x4 transformation matrix
def convert_matrix_to_vec(tf_matrix, mode='xyzw'):
	if mode == 'xyzw':
		# vec_p: position vector (x, y, z)
		# vec_q: quaternion vector (qx, qy, qz, qw)
		# Extract the translation vector from the matrix
		vec_p = tf_matrix[:3, 3]
		# Extract the rotation part of the matrix and convert it to a quaternion
		vec_q = Rotation.from_matrix(tf_matrix[:3, :3]).as_quat()
	if mode == 'wxyz':
		# vec_p: position vector (x, y, z)
		# vec_q: quaternion vector (qw, qx, qy, qz)
		# Extract the translation vector from the matrix
		vec_p = tf_matrix[:3, 3]
		# Extract the rotation part of the matrix and convert it to a quaternion
		vec_q = np.roll(Rotation.from_matrix(tf_matrix[:3, :3]).as_quat(), 1)
	return vec_p, vec_q

def compute_relative_dis(last_t, last_quat, curr_t, curr_quat, mode='xyzw'):
	if mode == 'xyzw':
		rot1 = Rotation.from_quat(last_quat)
		rot2 = Rotation.from_quat(curr_quat)
		rel_rot = rot2 * rot1.inv()
		dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
		dis_trans = np.linalg.norm(rot1.inv().apply(last_t - curr_t))
	if mode == 'wxyz':
		rot1 = Rotation.from_quat(np.roll(last_quat, -1))
		rot2 = Rotation.from_quat(np.roll(curr_quat, -1))   
		rel_rot = rot2 * rot1.inv()
		dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
		dis_trans = np.linalg.norm(rot1.inv().apply(last_t - curr_t))
	return dis_trans, dis_angle

def compute_relative_dis_TF(last_T, curr_T):
	rel_T = np.linalg.inv(last_T) @ curr_T
	rel_rot = Rotation.from_matrix(rel_T[:3, :3])
	dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
	dis_trans = np.linalg.norm(rel_T[:3, 3])
	return dis_trans, dis_angle
