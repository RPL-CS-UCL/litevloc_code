import torch
from PIL import Image
import torchvision.transforms as tfm
from pathlib import Path
from typing import Tuple

from matching import viz2d, get_matcher

def load_image(
    path: str | Path, resize: int | Tuple = None, rot_angle: float = 0
) -> torch.Tensor:
    if isinstance(resize, int):
        resize = (resize, resize)
    img = tfm.ToTensor()(Image.open(path).convert("RGB"))
    tensor_size1 = img.shape

    if resize is not None:
        img = tfm.Resize(resize, antialias=True)(img)
    img = tfm.functional.rotate(img, rot_angle)
    tensor_size2 = img.shape

    print(f' - adding {path} with resolution {tensor_size1} --> {tensor_size2}')
    return img


def save_visualization(image0, image1, mkpts0, mkpts1, out_dir, index, n_viz=1):
    """Save visualization of the matching results."""
    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(mkpts0[::n_viz], mkpts1[::n_viz], color="lime", lw=0.2)
    viz2d.add_text(0, f"{len(mkpts1)} matches", fs=20)
    viz_path = out_dir / f"output_{index}.jpg"
    viz2d.save_plot(viz_path)
    return viz_path

def save_output(result, img0_path, img1_path, matcher_name, n_kpts, im_size, out_dir, index):
    """Save the output data to a file."""
    dict_path = out_dir / f"output_{index}.torch"
    output_dict = {
        "num_inliers": result["num_inliers"],
        "H": result["H"],
        "mkpts0": result["inliers0"],
        "mkpts1": result["inliers1"],
        "img0_path": img0_path,
        "img1_path": img1_path,
        "matcher": matcher_name,
        "n_kpts": n_kpts,
        "im_size": im_size,
    }
    torch.save(output_dict, dict_path)
    return dict_path

def initialize_matcher(matcher, device, n_kpts):
    """Initialize the matcher with provided arguments."""
    return get_matcher(matcher, device=device, max_num_keypoints=n_kpts)

def matching_image_pair(matcher, image0, image1):
    """Process a pair of images using the matcher."""
    result = matcher(image0, image1)
    return result
