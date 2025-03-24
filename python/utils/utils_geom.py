# utils_geom.py
"""Geometric utility functions for SLAM and 3D vision tasks."""

import os
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Optional, Tuple

# Constants
DEFAULT_IMAGE_TEMPLATE = "seq/{frame_id:06d}.color.jpg"
QUAT_MODES = ('xyzw', 'wxyz')

def read_timestamps(file_path: str) -> Dict[str, float]:
    """Read timestamps from a file into a dictionary.
    
    Args:
        file_path: Path to timestamp file
        
    Returns:
        Dictionary mapping image names to timestamps
    """
    return _read_generic_file(file_path, data_dim=1)

def read_poses(file_path: str) -> Dict[str, np.ndarray]:
    """Read camera poses (quaternion + translation) from file.
    
    Args:
        file_path: Path to pose file
        
    Returns:
        Dictionary mapping image names to pose vectors (qw, qx, qy, qz, tx, ty, tz)
    """
    return _read_generic_file(file_path, data_dim=7)

def read_intrinsics(file_path: str) -> Dict[str, np.ndarray]:
    """Read camera intrinsics from file.
    
    Args:
        file_path: Path to intrinsics file
        
    Returns:
        Dictionary mapping image names to intrinsic parameters [fx, fy, cx, cy, w, h]
    """
    return _read_generic_file(file_path, data_dim=6)

def read_descriptors(file_path: str) -> Dict[str, np.ndarray]:
    """Read feature descriptors from file.
    
    Args:
        file_path: Path to descriptor file
        
    Returns:
        Dictionary mapping image names to descriptor vectors
    """
    return _read_generic_file(file_path, data_dim=None)

def _read_generic_file(file_path: str, data_dim: Optional[int]) -> Dict[str, np.ndarray]:
    """Generic file reader helper function."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}

    data_dict = {}
    with open(file_path, 'r') as f:
        for line_id, line in enumerate(f):
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            if parts[0].startswith('seq'):
                img_name = parts[0]
                data = list(map(float, parts[1:]))
            else:
                img_name = DEFAULT_IMAGE_TEMPLATE.format(frame_id=line_id)
                data = list(map(float, parts))

            if data_dim and len(data) != data_dim:
                print(f"Ignoring malformed line {line_id}: {line.strip()}")
                continue

            data_dict[img_name] = np.array(data)
            
    return data_dict

def convert_vec_gtsam_pose3(
    translation, 
    quaternion, 
    mode='xyzw'
) -> gtsam.Pose3:
    if mode not in QUAT_MODES:
        raise ValueError(f"Invalid quaternion mode: {mode}")

    if mode == 'xyzw':
        quaternion = np.roll(quaternion, -1)

    pose3 = gtsam.Pose3(gtsam.Rot3(quaternion), translation.reshape(3, 1))
    return pose3

def convert_vec_to_matrix(
    translation: np.ndarray,
    quaternion: np.ndarray,
    mode: str = 'xyzw'
) -> np.ndarray:
    """Convert translation and quaternion to 4x4 transformation matrix.
    
    Args:
        translation: [x, y, z] translation vector
        quaternion: Quaternion components
        mode: Quaternion format ('xyzw' or 'wxyz')
        
    Returns:
        4x4 transformation matrix
    """
    tf = np.eye(4)
    tf[:3, 3] = translation
    
    if mode not in QUAT_MODES:
        raise ValueError(f"Invalid quaternion mode: {mode}")
        
    if mode == 'wxyz':
        quaternion = np.roll(quaternion, -1)
        
    tf[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return tf

def convert_matrix_to_vec(
    transform: np.ndarray,
    mode: str = 'xyzw'
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 transformation matrix to translation and quaternion.
    
    Args:
        transform: 4x4 transformation matrix
        mode: Desired quaternion format ('xyzw' or 'wxyz')
        
    Returns:
        Tuple of (translation, quaternion)
    """
    if transform.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix")
        
    translation = transform[:3, 3]
    rotation = Rotation.from_matrix(transform[:3, :3])
    quat = rotation.as_quat()
    
    if mode == 'wxyz':
        quat = np.roll(quat, 1)
        
    return translation, quat

def compute_pose_error(
    pose1: np.ndarray,
    pose2: np.ndarray,
    mode: str = 'matrix'
) -> Tuple[float, float]:
    """Compute relative pose error between two transformations.
    
    Args:
        pose1: First pose (4x4 matrix or translation+quaternion)
        pose2: Second pose (same format as pose1)
        mode: Input format ('matrix' or 'vector')
        
    Returns:
        Tuple of (translation_error [m], rotation_error [deg])
    """
    if mode == 'matrix':
        return _compute_error_from_matrices(pose1, pose2)
    elif mode == 'vector':
        return _compute_error_from_vectors(*pose1, *pose2)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def _compute_error_from_matrices(
    mat1: np.ndarray,
    mat2: np.ndarray
) -> Tuple[float, float]:
    """Compute error between two transformation matrices."""
    rel_tf = np.linalg.inv(mat1) @ mat2
    rot_error = Rotation.from_matrix(rel_tf[:3, :3]).magnitude() * 180/np.pi
    trans_error = np.linalg.norm(rel_tf[:3, 3])
    return trans_error, rot_error

def _compute_error_from_vectors(
    trans1: np.ndarray,
    quat1: np.ndarray,
    trans2: np.ndarray,
    quat2: np.ndarray,
    mode: str = 'xyzw'
) -> Tuple[float, float]:
    """Compute error between two pose vectors."""
    tf1 = convert_vec_to_matrix(trans1, quat1, mode)
    tf2 = convert_vec_to_matrix(trans2, quat2, mode)
    return _compute_error_from_matrices(tf1, tf2)

if __name__ == "__main__":
    # Example usage
    tf = convert_vec_to_matrix(
        translation=[1, 2, 3],
        quaternion=[0, 0, 0, 1],  # Identity rotation
        mode='xyzw'
    )
    print("Transformation matrix:\n", tf)
    
    trans, quat = convert_matrix_to_vec(tf, mode='xyzw')
    print("Recovered translation:", trans)
    print("Recovered quaternion:", quat)
