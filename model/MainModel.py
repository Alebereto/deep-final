import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.notebook import tqdm

import os
from shutil import copyfile

from model.Discriminator import Discriminator
from model.GANLoss import GANLoss
from model.Unet import Unet
from model.logger import Logger

from data.ImagesDataset import tensor_to_image


SAVE_PATH = "results"

class Painter(nn.Module):
	""" GAN architecture with Unet generator """

	def __init__(self, name:str, hyparams, load=False, load_pretrain=False, device=None):
		super(Painter, self).__init__()

		self.name = name
		self.device = device

		self.gan_criterion = GANLoss(device)
		self.l1_criterion = nn.L1Loss()

		if load: self.load(load_pretrain)
		else:
			self.logger = Logger(name)

			self.generator = Unet().to(self.device)
			self.discriminator = Discriminator().to(self.device)

			self.generator.apply(weights_init)
			self.discriminator.apply(weights_init)

		self.pre_optimizer = torch.optim.Adam(self.generator.parameters(), lr=hyparams.lr_pre)
		self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=hyparams.lr_g, betas=(0.5,0.999))
		self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=hyparams.lr_d, betas=(0.5,0.999))


	def paint(self, gray_image):
		""" Gets grayscale image, returns painted image """
		
		self.generator.eval()
		gray_tensor = torch.tensor([gray_image], dtype=torch.float32, device=self.device)
		lab_image = torch.cat((gray_tensor, self.generator(gray_tensor)), dim=0)
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
		loss = self.gan_criterion(fake_preds, real_data=True)
		loss.backward()

		self.optimizer_g.step()
		return loss.item()
	

	def train_model(self, trainloader) -> tuple[float,float]:
		""" Trains model for 1 epoch, returns average loss for generator and discriminator """

		self.generator.train()	# maybe needs to be eval when generating fake images for discriminator
		sum_g_loss = 0.
		sum_d_loss = 0.

		for l_batch, ab_batch in tqdm(trainloader, desc='Train'):

			# get real and fake images
			real_images = torch.cat((l_batch, ab_batch), dim=1)
			fake_ab_batch = self.generator(l_batch)
			fake_images = torch.cat((l_batch, fake_ab_batch), dim=1)
			
			sum_d_loss += self.optimize_discriminator(real_images, fake_images.detach())
			sum_g_loss += self.optimize_generator(fake_images)

		return sum_g_loss / len(trainloader), sum_d_loss / len(trainloader)

	def test_model(self, testloader, pretrain=False):
		""" Tests model on testset, returns average loss for generator and discriminator """

		self.generator.eval()
		self.discriminator.eval()
		sum_g_loss = 0.
		sum_d_loss = 0.

		with torch.no_grad():
			for l_batch, ab_batch in tqdm(testloader, desc='Test'):
				real_images = torch.cat((l_batch, ab_batch), dim=1)
				fake_ab_batch = self.generator(l_batch)
				fake_images = torch.cat((l_batch, fake_ab_batch), dim=1)

				# save some images from batch to logger
				self.logger.add_images(real_images, fake_images)

				# get losses
				if pretrain:
					sum_g_loss += self.l1_criterion(fake_images, real_images).item()
				else:
					fake_preds = self.discriminator(fake_images)
					real_preds = self.discriminator(real_images)

					g_loss = self.gan_criterion(fake_preds, real_data=True)
					sum_g_loss += g_loss.item()

					loss_fake = self.gan_criterion(fake_preds, real_data=False)
					loss_real = self.gan_criterion(real_preds, real_data=True)

					loss = (loss_fake + loss_real) * 0.5
					sum_d_loss += loss.item()

		if pretrain: losses = sum_g_loss / len(testloader)
		else: losses = (sum_g_loss / len(testloader), sum_d_loss / len(testloader))

		return losses

	def pretrain_generator(self, trainloader) -> None:
		self.generator.train()
		sum_loss = 0

		for l_batch, ab_batch in tqdm(trainloader, desc='Train'):

			# get real and fake images
			real_images = torch.cat((l_batch, ab_batch), dim=1)
			fake_ab_batch = self.generator(l_batch)
			fake_images = torch.cat((l_batch, fake_ab_batch), dim=1)
			
			loss = self.l1_criterion(fake_images, real_images)
			self.pre_optimizer.zero_grad()
			loss.backward()
			self.pre_optimizer.step()

			sum_loss += loss.item()

		return sum_loss / len(trainloader)

	def save(self, pretrain=False):
		data_name = 'save_data_pre.pt' if pretrain else 'save_data.pt'
		model_path = os.path.join(SAVE_PATH, self.name)
		if not os.path.isdir(model_path): os.mkdir(model_path)
		save_data = (self.logger, self.generator.state_dict(), self.discriminator.state_dict())

		if self.logger.epochs_pretrained > 0: self.logger.plot_performence(show=False, pretrain=True)
		if self.logger.epochs_trained > 0: self.logger.plot_performence(show=False)
		if len(self.logger.recent_images) > 0: self.logger.plot_coloring(show=False)
		torch.save(save_data, os.path.join(model_path, data_name))

	def load(self, pretrain=False):
		data_name = 'save_data_pre.pt' if pretrain else 'save_data.pt'
		model_path = os.path.join(SAVE_PATH, self.name)
		assert os.path.isdir(model_path), f'Model with name "{self.name}" does not exist in path'

		logger, g_weights, d_weights = torch.load(os.path.join(model_path, data_name))
		self.logger = logger
		logger.name = self.name

		self.generator = Unet().to(self.device)
		self.generator.load_state_dict(g_weights)
		self.discriminator = Discriminator().to(self.device)
		if not pretrain: self.discriminator.load_state_dict(d_weights)


	def count_parameters(self):
		g_pcount = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
		self.discriminator.set_gradients(True)
		d_pcount = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

		return g_pcount + d_pcount

def weights_init(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and 'Conv' in classname:
			nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
		elif 'BatchNorm2d' in classname:
			nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
			nn.init.constant_(m.bias.data, 0.)

