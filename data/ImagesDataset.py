import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
import os
import random

from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import cv2
import warnings


class ImagesDataset(Dataset):
	def __init__(self, data_paths: list[str], foods:list[str], device):
		self.paths = data_paths
		self.foods = foods
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
		print(f'Food count: {len(self.foods)}')
		print(f'Image count: {len(self.paths)}')
		print()


def create_datasets(data_path: str, train_size: int, test_size: int, n_labels=25, food_list=None, seed=None, device=None) -> tuple[ImagesDataset, ImagesDataset]:
	""" Returns train and test datasets """

	if seed is not None: random.seed(seed)

	if food_list is None:
		all_foods = os.listdir(data_path)
		foods = random.choices(all_foods, k=n_labels)
	else:
		with open('data\\food_list.txt', 'r') as file:
			foods = file.read().split('\n')
		n_labels = len(foods)

	img_paths_train = list()
	img_paths_test = list()

	train_count = train_size // n_labels
	test_count = test_size // n_labels

	assert train_count + test_count <= 500, "Not enough data for specified sizes"

	for food in foods:
		dr = f'{data_path}\\{food}'
		paths = glob(f'{dr}\\*.jpg')
		random.shuffle(paths)
		train_paths, test_paths = paths[:train_count], paths[(len(paths)-test_count):]
		for path in train_paths: img_paths_train.append(path)
		for path in test_paths:  img_paths_test.append(path)

	return ImagesDataset(img_paths_train, foods, device), ImagesDataset(img_paths_test, foods, device)

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
	""" gets grayscale image, returns notmalized lab tensor """

	toTensor = transforms.ToTensor()

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	img = rgb2lab(np.array(img)).astype(np.float32)
	img = toTensor(img)
	l = (img[[0],...] / 50.) -1.	# shape (1,H,W)
	return l

def add_noise(tensor: torch.Tensor, std=0.04, mean=0.) -> torch.Tensor:
	noised = tensor + torch.randn(tensor.size(), device=(tensor.get_device())) * std + mean
	return torch.clamp(noised, min=-1, max=1)

