import numpy as np
import os
from glob import glob
import argparse

def load_and_concat_npy(input_files, axis=0):
    """
    Load and concatenate multiple .npy files along specified axis
    
    Parameters:
    input_files (str/list): Directory path, glob pattern, or list of file paths
    axis (int): Axis along which to concatenate (default: 0)
    
    Returns:
    np.ndarray: Concatenated array
    
    Example:
    combined = load_and_concat_npy("data/*.npy")
    """
    # Handle different input types
    if isinstance(input_files, str):
        files = sorted(glob(input_files))
    else:
        raise ValueError("Input must be directory path, glob pattern, or list of files")

    if not files:
        raise FileNotFoundError("No .npy files found in the specified path")

    # Load first array to get reference shape
    arrays = []
    try:
        first_arr = np.load(files[0])
        ref_shape = list(first_arr.shape)
        ref_shape.pop(axis)
        arrays.append(first_arr)
    except Exception as e:
        raise IOError(f"Error loading {files[0]}") from e

    # Load remaining arrays with shape validation
    for f in files[1:]:
        try:
            arr = np.load(f)
            # Verify compatible shape (ignore concatenation axis)
            test_shape = list(arr.shape)
            test_shape.pop(axis)
            if test_shape != ref_shape:
                raise ValueError(f"Shape mismatch in {f}: expected {ref_shape} (excluding axis {axis}), got {test_shape}")
            arrays.append(arr)
        except Exception as e:
            raise IOError(f"Error loading {f}") from e

    return np.concatenate(arrays, axis=axis)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge depth files with specified suffix')
    parser.add_argument('--dataset_dir', type=str, 
                       help='Dataset directory path')
    parser.add_argument('--depth_suffix', type=str, default='pdepth',
                       help='Suffix for depth files (default: pdepth, gtdepth, m3ddepth)')
    args = parser.parse_args()
    
    input_pattern = f"{args.dataset_dir}/pairs/mapfree_pairs_s*{args.depth_suffix}.npy"
    output_file = f"{args.dataset_dir}/pairs/mapfree_pairs_{args.depth_suffix}.npy"
    
    combined = load_and_concat_npy(input_pattern)
    np.save(output_file, combined)
    print(f"Combined array shape: {combined.shape}")

