import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		input = 3
		filters = 32
		self.main = nn.Sequential(
			nn.Conv2d(input, filters, 6, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(filters, (filters * 2), 6, 2, 1, bias=False),
			nn.BatchNorm2d(filters * 2),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(filters * 2, (filters * 4), 6, 2, 1, bias=False),
			nn.BatchNorm2d(filters * 4),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(filters * 4, (filters * 8), 6, 2, 1, bias=False),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(filters * 8, (filters * 8), 6, 2, 1, bias=False),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d((filters * 8), 1, 6, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		output = self.main(input).view(-1)
		return output
	
	def set_gradients(self, grad: bool):
		for p in self.parameters():
			p.requires_grad = grad

