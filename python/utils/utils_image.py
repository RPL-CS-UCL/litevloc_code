import torch
import torchvision.transforms as tfm
from PIL import Image
from pathlib import Path
from typing import Tuple
import logging

def load_image(
	path: str | Path, resize: int | Tuple = None, rot_angle: float = 0,
	normalized: bool = False
) -> torch.Tensor:
	if isinstance(resize, int):
		resize = (resize, resize)
	transformations = [tfm.ToTensor()]
	if normalized:
		transformations.append(
			tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	if resize:
		transformations.append(
			tfm.Resize(size=resize, antialias=True))
	transform = tfm.Compose(transformations)

	pil_img = Image.open(path).convert("RGB")
	tensor_size1 = (pil_img.size[1], pil_img.size[0])
	
	img = transform(pil_img)
	tensor_size2 = img.shape

	logging.debug(f' - adding {path} with resolution {tensor_size1} --> {tensor_size2}')
	return img
