from typing import Union, Tuple
from pathlib import Path
import torch
from torchvision import transforms as tfm
from PIL import Image
import numpy as np
import logging

def load_rgb_image(
    path: Union[str, Path],
    resize: Union[int, Tuple] = None,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Load an RGB image from the given path and apply transformations.

    Args:
        path (Union[str, Path]): The path to the image file.
        resize (Union[int, Tuple], optional): The desired size of the image. 
        normalized (bool, optional): Whether to normalize the image. If True, the image will be normalized using the mean and standard deviation values specified in the code. Defaults to False.

    Returns:
        torch.Tensor: The transformed image as a tensor (3, H, W)

    """
    if isinstance(resize, int):
        resize = (resize, resize)
    
    # Set up transformations: - Convert to tensor, - Normalize, - Resize
    transformations = [tfm.ToTensor()]
    if normalized:
        transformations.append(
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    if resize is not None:
        new_size = (resize[1], resize[0]) # HxW
        transformations.append(tfm.Resize(size=new_size, antialias=True))
    transform = tfm.Compose(transformations)

    # Load image and apply transformation
    pil_img = Image.open(path).convert("RGB")
    tensor_size1 = (pil_img.size[1], pil_img.size[0])

    img = transform(np.array(pil_img))
    tensor_size2 = img.shape

    logging.debug(f" - adding {path} with resolution {tensor_size1} --> {tensor_size2}")
    return img

def load_depth_image(
    path: Union[str, Path],
    depth_scale=0.001,
) -> torch.Tensor:
    # Set up transformations: - Convert to tensor
    transformations = [tfm.ToTensor()]
    transform = tfm.Compose(transformations)

    # Load image and apply transformation
    pil_img = Image.open(path)
    tensor_size1 = (pil_img.size[1], pil_img.size[0])
    img = transform(np.array(pil_img) * depth_scale)
    tensor_size2 = img.shape

    logging.debug(f" - adding {path} with resolution {tensor_size1} --> {tensor_size2}")
    return img

def rgb_image_to_tensor(
    rgb_img: np.ndarray,
    resize: Union[int, Tuple] = None,
    normalized: bool = False,
) -> torch.Tensor:
    if isinstance(resize, int):
        resize = (resize, resize)
    # Set up transformations: - Convert to tensor, - Normalize, - Resize
    transformations = [tfm.ToTensor()]
    if normalized:
        transformations.append(
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    if resize is not None:
        new_size = (resize[1], resize[0]) # width, height
        transformations.append(tfm.Resize(size=new_size, antialias=True))
    transform = tfm.Compose(transformations)

    # Load image and apply transformation
    tensor_size1 = (rgb_img.shape[1], rgb_img.shape[0])

    rgb_img_copy = rgb_img.copy()
    img = transform(rgb_img_copy)
    tensor_size2 = img.shape

    logging.debug(f" - adding rgb image with resolution {tensor_size1} --> {tensor_size2}")
    return img

def depth_image_to_tensor(
    depth_img: np.ndarray,
    depth_scale=0.001,
) -> torch.Tensor:
    # Set up transformations: - Convert to tensor
    transformations = [tfm.ToTensor()]
    transform = tfm.Compose(transformations)

    # Load image and apply transformation
    tensor_size1 = (depth_img.shape[1], depth_img.shape[0])

    depth_img_copy = (depth_img * depth_scale).copy()
    img = transform(depth_img_copy)
    tensor_size2 = img.shape

    logging.debug(f" - adding depth image with resolution {tensor_size1} --> {tensor_size2}")
    return img
