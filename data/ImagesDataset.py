import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
from shutil import copyfile
import os
import random

from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import cv2
import warnings


class ImagesDataset(Dataset):
	def __init__(self, data_paths: list[str], device):
		self.paths = data_paths
		self.device = device
		self.transform = transforms.Resize((256, 256),  Image.BICUBIC)
		self.toTensor = transforms.ToTensor()

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
		""" Returns normalized (-1 to 1) L channel with shape [1,H,W],
			and normalized (-1 to 1) ab channels with shape [2,H,W] """

		img = Image.open(self.paths[idx]).convert("RGB")
		img = np.array(self.transform(img))
		img = rgb2lab(img).astype(np.float32)		# convert to lab (and lower from float64 to float32)
		img = self.toTensor(img).to(self.device)	# transform to tensor (also changes shape to C*H*W)
		
		l = (img[[0],...] / 50.) -1.	# Get normalized L channel (value range is (0, 100))
		ab = img[[1,2],...] / 110.	# Get normalized ab channels (value range is (-107.8573, 100))

		return l, ab
	
	def print_stats(self):
		print('=====Dataset Stats=====')
		print(f'Image count: {len(self.paths)}')
		print()


def create_datasets(data_path: str, train_size: int, test_size: int, preset=False, seed=None, device=None) -> tuple[ImagesDataset, ImagesDataset]:
	""" Returns train and test datasets """

	if preset:
		train_paths = glob(f'{data_path}\\train\\*.jpg')
		test_paths = glob(f'{data_path}\\test\\*.jpg')

	else:
		DATA_NAME = "food"

		paths = glob(f'{data_path}\\{DATA_NAME}\\**\\*.jpg') # get paths to all images (jpg)

		assert train_size + test_size < len(paths), "Not enough data for specified sizes"

		if seed is not None: random.seed(seed)
		random.shuffle(paths)
		train_paths, test_paths = paths[:train_size], paths[(len(paths)-test_size):]

	return ImagesDataset(train_paths, device), ImagesDataset(test_paths, device)

def tensor_to_image(tensor:torch.Tensor) -> np.ndarray:
	""" gets normalized lab image as tensor, returns rgb image as numpy array """

	l = (tensor[[0],...] + 1.) * 50.	# Un-normalize
	ab = tensor[[1,2],...] * 110.	    	# Un-normalize
	tensor = torch.cat([l,ab], dim=0)

	img = tensor.permute(1,2,0).cpu().numpy()	# reshape to (H,W,C)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		img = lab2rgb(img) * 255	# (result is values from 0 to 1)
	return np.clip(img, a_min=0, a_max=255).astype(np.uint8)

def gray_to_tensor(img) -> torch.Tensor:
	""" gets grayscale image, returns normalized lab tensor """

	toTensor = transforms.ToTensor()

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	img = rgb2lab(np.array(img)).astype(np.float32)
	img = toTensor(img)
	l = (img[[0],...] / 50.) -1.	# shape (1,H,W)
	return l

def add_noise(tensor: torch.Tensor, std=0.04, mean=0.) -> torch.Tensor:
	noised = tensor + torch.randn(tensor.size(), device=(tensor.device)) * std + mean
	return torch.clamp(noised, min=-1, max=1)

