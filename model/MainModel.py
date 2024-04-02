import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.notebook import tqdm

import os
from shutil import copyfile

from network import Unet, Discriminator, Loss
from logger import Logger


class Painter(nn.Module):
	def __init__(self, hyparams=None, g_params=None, d_params=None, load_path=None):


		self.logger = Logger(hyparams, g_params, d_params)

		self.generator = Unet(g_params)
		self.discriminator = Discriminator(d_params)

		self.gan_criterion = Loss()
		self.l1_criterion = nn.L1Loss()

		self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=hyparams.lr_g)
		self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=hyparams.lr_d)


	def train_model(self, trainloader):
		self.generator.train()
		self.discriminator.train()

		for l_batch, ab_batch in tqdm(trainloader):	# Note: tdqm is a wrapper that shows progress (didnt try it yet)

			# set data to device
			l_batch = l_batch.to(self.device)
			ab_batch = ab_batch.to(self.device)

			self.optimizer_g.zero_grad()

			# forward
			# ?????????
			fake_ab = self.generator(l_batch)
			# backwards
			# ????????

			# log accuracy
			# ??????

			# if (self.global_batch % 30) == 0:
			# 	print(f"batch {self.global_batch:>4d}, loss = {loss.item():.4f}")


	def test_model(self, testloader, prints=True):
		pass


	def save_model(self, backup=True):
		pass

	def load_model(self, path):
		pass

	def count_parameters(self):
		# maybe does not work
		return sum(p.numel() for p in self.gan() if p.requires_grad)

