from typing import Union, Tuple
from pathlib import Path
import torch
from torchvision import transforms as tfm
from PIL import Image
import numpy as np
import logging

class ColorCorrection:
    """
    Torch-compatible transform that combines Gray World correction with:
    - Gamma-aware processing (sRGB <-> linear)
    - Daylight (5000K) color temperature compensation
    - Blue channel reduction to counteract Aria's blueish tint
    """
    @staticmethod
    def srgb_to_linear(ts: torch.Tensor) -> torch.Tensor:
        """Convert sRGB to linear RGB (gamma decoding)."""
        return torch.where(ts <= 0.04045, ts / 12.92, ((ts + 0.055) / 1.055).pow(2.4))

    @staticmethod
    def linear_to_srgb(ts: torch.Tensor) -> torch.Tensor:
        """Convert linear RGB to sRGB (gamma encoding)."""
        return torch.where(ts <= 0.0031308, ts * 12.92, 1.055 * ts.pow(1 / 2.4) - 0.055)

    def __init__(self, comp_blue: float = 0.95):
        self.temp_comp = torch.tensor([
            [1.00, 0.00, 0.00],      # Red stays same
            [0.00, 1.00, 0.00],      # Green stays same
            [0.00, 0.00, comp_blue]  # Reduce blue slightly
        ])

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # Step 1: sRGB -> linear RGB
        linear = ColorCorrection.srgb_to_linear(img_tensor)
        # Step 2: Gray World correction in linear RGB
        means = linear.mean(dim=[1, 2])             # Channel means
        mean_intensity = means.mean()
        scale = mean_intensity / (means + 1e-6)     # Avoid divide-by-zero
        scaled = linear * scale[:, None, None]      # Apply scale per channel
        # Step 3: Apply 5000K temp compensation matrix
        corrected = torch.einsum("ij,jhw->ihw", self.temp_comp, scaled)
        # Step 4: linear -> sRGB
        srgb = ColorCorrection.linear_to_srgb(torch.clamp(corrected, 0, 1))

        return srgb

def load_rgb_image(
    path: Union[str, Path],
    resize: Union[int, Tuple] = None,
    normalized: bool = False,
    color_correct: bool = False
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
    
    # Set up transformations: - Correct Color Space - Convert to tensor - Normalize, - Resize
    transformations = [tfm.ToTensor()]
     
    if normalized:
        transformations.append(
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    if color_correct:
        transformations.append(ColorCorrection(comp_blue=0.95))

    if resize is not None:
        new_size = (resize[1], resize[0]) # HxW
        transformations.append(tfm.Resize(size=new_size, antialias=True))

    transform = tfm.Compose(transformations)

    # Load image and apply transformation
    pil_img = Image.open(path).convert("RGB")
    # tensor_size1 = (pil_img.size[1], pil_img.size[0])
    img = transform(np.array(pil_img))
    # tensor_size2 = img.shape

    # logging.debug(f" - adding {path} with resolution {tensor_size1} --> {tensor_size2}")
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
    # tensor_size1 = (pil_img.size[1], pil_img.size[0])
    img = transform(np.array(pil_img) * depth_scale)
    # tensor_size2 = img.shape

    # logging.debug(f" - adding {path} with resolution {tensor_size1} --> {tensor_size2}")
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
