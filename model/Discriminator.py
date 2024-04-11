import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
	def __init__(self) -> None:
		super(Discriminator, self).__init__()
		self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
			# nn.BatchNorm2d(),
			nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
		self.linear = nn.Linear(11111, 1)	# need flattened dimension

	def forward(self, x):
		x = self.conv(x)
		print(x.size())
		x = x.view(-1, 23123123) # need flattened dimension
		x = self.linear(x)
		return F.sigmoid(x)

	def set_gradients(self, grad: bool):
		for p in self.parameters():
			p.requires_grad = grad

