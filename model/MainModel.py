import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.notebook import tqdm

import os
from shutil import copyfile

from network import Unet, Discriminator, GANLoss
from logger import Logger

from data.ImagesDataset import tensor_to_image


SAVE_PATH = "results"

class Painter(nn.Module):
	def __init__(self, name:str, hyparams=None, g_params=None, d_params=None, load=False):
		self.name = name

		self.gan_criterion = GANLoss()
		# self.l1_criterion = nn.L1Loss()

		if load: self.load()
		else:
			self.logger = Logger(name, hyparams, g_params, d_params)

			self.generator = Unet(g_params)
			self.discriminator = Discriminator(d_params)

			self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=hyparams.lr_g)
			self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=hyparams.lr_d)

	def paint(self, gray_image):
		""" Gets grayscale image, returns painted image """
		
		self.generator.eval()
		gray_tensor = torch.tensor([gray_image], dtype=torch.float32)
		lab_image = torch.cat(gray_tensor, self.generator(gray_tensor), dim=0)
		return tensor_to_image(lab_image)


	def optimize_discriminator(self, real_images, fake_images) -> float:
		""" Do a gradient step for discriminator, return loss """

		self.discriminator.set_gradients(True)
		self.discriminator.train()
		self.optimizer_d.zero_grad()
		
		fake_preds = self.discriminator(fake_images)
		loss_fake = self.gan_criterion(fake_preds, real_data=False)

		real_preds = self.discriminator(real_images)
		loss_real = self.gan_criterion(real_preds, real_data=True)

		loss = (loss_fake + loss_real) * 0.5
		loss.backward()
		
		self.optimizer_d.step()
		return loss.item()
	
	def optimize_generator(self, fake_images) -> float:
		""" Do a gradient step for generator, return loss """

		self.discriminator.set_gradients(False)
		self.discriminator.eval()
		self.optimizer_g.zero_grad()

		fake_preds = self.discriminator(fake_images)
		loss = self.gan_criterion(fake_preds, True)
		loss.backward()

		self.optimizer_g.step()
		return loss.item()
	

	def train_model(self, trainloader) -> tuple[float,float]:
		""" Trains model for 1 epoch, returns average loss for generator and discriminator """

		self.generator.train()	# maybe needs to be eval when generating fake images for discriminator
		sum_g_loss = 0
		sum_d_loss = 0

		for l_batch, ab_batch in tqdm(trainloader):	# Note: tdqm is a wrapper that shows progress (didnt try it yet)

			# set data to device
			l_batch = l_batch.to(self.device)
			ab_batch = ab_batch.to(self.device)

			# get real and fake images
			real_images = torch.cat(l_batch, ab_batch, dim=1)
			fake_ab_batch = self.generator(l_batch)
			fake_images = torch.cat(l_batch, fake_ab_batch, dim=1)

			
			sum_d_loss += self.optimize_discriminator(real_images, fake_images.detach())
			sum_g_loss += self.optimize_generator(fake_images)

		return sum_g_loss / len(trainloader), sum_d_loss / len(trainloader)

	def test_model(self, testloader) -> tuple[float,float]:
		""" Tests model on testset, returns average loss for generator and discriminator """

		self.generator.eval()
		self.discriminator.eval()
		sum_g_loss = 0
		sum_d_loss = 0

		with torch.no_grad():
			for l_batch, ab_batch in testloader:
				real_images = torch.cat(l_batch, ab_batch, dim=1)
				fake_ab_batch = self.generator(l_batch)
				fake_images = torch.cat(l_batch, fake_ab_batch, dim=1)

				# save some images from batch to logger
				self.logger.add_images(real_images, fake_images)

				# TODO: get losses and add to sum
				sum_g_loss += 0
				sum_d_loss += 0

		return sum_g_loss / len(testloader), sum_d_loss / len(testloader)

	def save(self):
		model_path = os.path.join(SAVE_PATH, self.name)
		if not os.path.isdir(model_path): os.mkdir(model_path)
		save_data = (self.logger, self.generator.state_dict(), self.discriminator.state_dict())

		self.logger.plot_performence(show=False)
		self.logger.plot_coloring(show=False)
		torch.save(save_data, os.path.join(model_path, 'save_data.pt'))

	def load(self):
		model_path = os.path.join(SAVE_PATH, self.name)
		assert os.path.isdir(model_path), f'Model with name "{self.name}" does not exist in path'

		logger, g_weights, d_weights = torch.load(os.path.join(model_path, 'save_data.pt'))
		self.logger = logger

		self.generator = Unet(logger.g_params)
		self.generator.load_state_dict(g_weights)
		self.discriminator = Discriminator(logger.d_params)
		self.discriminator.load_state_dict(d_weights)

		self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=logger.hyparams.lr_g)
		self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=logger.hyparams.lr_d)


	def count_parameters(self):
		g_pcount = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
		self.discriminator.set_gradients(True)
		d_pcount = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

		return g_pcount + d_pcount

