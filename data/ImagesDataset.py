import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
import os

from cv2 import imread
from skimage.color import rgb2lab, lab2rgb


class ImagesDataset(Dataset):
	def __init__(self, data_paths: list[str]):
		self.paths = data_paths

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
		""" Returns normalized (-1 to 1) L channel with shape [1,H,W],
			and normalized (-1 to 1) ab channels with shape [2,H,W] """

		image = imread(self.paths[idx])	# load image as rgb
		# TODO: maybe add noise if train dataset
		# TODO: resize image to be uniform shape
		image_lab = rgb2lab(image).astype(np.float32)	# convert to lab (and lower from float64 to float32)
		tensor_image = transforms.ToTensor(image_lab)	# transform to tensor (also changes shape to C*H*W)
		
		l = tensor_image[[0],...] / 50. -1.	# Get normalized L channel (value range is (0, 100))
		ab = tensor_image[[1,2],...] / 110.	# Get normalized ab channels (value range is (-107.8573, 100))

		return l, ab


def create_datasets(data_path: str, train_size: int, seed=None) -> tuple[ImagesDataset, ImagesDataset]:
	""" Returns train and test datasets, split with train_size """

	paths = glob(os.path.join(data_path, '*.jpg'))	# get paths to all images (jpg)
	assert train_size < len(paths), "Train size too big (no test set)"

	if seed is not None: np.random.seed(seed)
	indeces = np.random.permutation(len(paths))
	train_paths, test_paths = paths[indeces[:train_size]], paths[indeces[train_size:]]

	return ImagesDataset(train_paths), ImagesDataset(test_paths)

def tensor_to_image(tensor:torch.Tensor) -> np.ndarray:
	""" gets lab image as tensor, returns rgb image as numpy array """

	lab_image = tensor.permute(1,2,0).numpy()
	return lab2rgb(lab_image)

def gray_tensor_to_image(tensor:torch.Tensor) -> np.ndarray:
	""" gets grayscale image as tensor, returns grayscale image as numpy array """

	return tensor.numpy()[0]

