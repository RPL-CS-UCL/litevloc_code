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
    rot_angle: float = 0,
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
        transformations.append(tfm.Resize(size=resize, antialias=True))
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
    resize: Union[int, Tuple] = None,
    rot_angle: float = 0,
    depth_scale=0.001,
) -> torch.Tensor:
    if isinstance(resize, int):
        resize = (resize, resize)
    # Set up transformations: - Convert to tensor, - Resize
    transformations = [tfm.ToTensor()]
    if resize is not None:
        transformations.append(tfm.Resize(size=resize, antialias=True))
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
    rot_angle: float = 0,
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
        transformations.append(tfm.Resize(size=resize, antialias=True))
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
    resize: Union[int, Tuple] = None,
    rot_angle: float = 0,
    depth_scale=0.001,
) -> torch.Tensor:
    if isinstance(resize, int):
        resize = (resize, resize)
    # Set up transformations: - Convert to tensor, - Resize
    transformations = [tfm.ToTensor()]
    if resize is not None:
        transformations.append(tfm.Resize(size=resize, antialias=True))
    transform = tfm.Compose(transformations)

    # Load image and apply transformation
    tensor_size1 = (depth_img.shape[1], depth_img.shape[0])

    depth_img_copy = (depth_img * depth_scale).copy()
    img = transform(depth_img_copy)
    tensor_size2 = img.shape

    logging.debug(f" - adding depth image with resolution {tensor_size1} --> {tensor_size2}")
    return img
