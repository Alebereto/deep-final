import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Unet(nn.Module):
	def __init__(self, params) -> None:
		super(Unet, self).__init__()
		pass

	def forward(self, x):
		pass
	

class Discriminator(nn.Module):
	def __init__(self, params) -> None:
		super(Discriminator, self).__init__()
		pass

	def forward(self, x):
		pass

	def set_gradients(self, grad: bool):
		for p in self.parameters():
			p.requires_grad = grad


class GANLoss():
	def __init__(self) -> None:
		self.loss_func = nn.MSELoss()
	
	def __call__(self, pred, real_data: bool) -> torch.Tensor:
		if real_data: label = 1.
		else: label = 0.
		loss = self.loss_func(pred, label)
		return loss

