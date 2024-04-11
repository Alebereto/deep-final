import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
	def __init__(self, params) -> None:
		super(Discriminator, self).__init__()
		pass

	def forward(self, x):
		pass

	def set_gradients(self, grad: bool):
		for p in self.parameters():
			p.requires_grad = grad

