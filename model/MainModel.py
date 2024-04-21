import torch
import torch.nn as nn
import numpy as np

from tqdm.notebook import tqdm

import os

from model.Discriminator import Discriminator
from model.GANLoss import GANLoss
from model.Unet import Unet
from model.logger import Logger, LossGatherer

from data.ImagesDataset import tensor_to_image, gray_to_tensor
from PIL import Image
from torchvision import transforms


SAVE_PATH = "results"

class Painter(nn.Module):
	""" GAN architecture with Unet generator """

	def __init__(self, name:str, lr_d=1e-4, lr_g=2e-4, lr_pre=1e-4, load=False, load_pretrain=False, device=None):
		super(Painter, self).__init__()
		if device is None: device = torch.device('cpu')

		self.name = name
		self.device = device

		self.gan_criterion = GANLoss(device)
		self.l1_criterion = nn.L1Loss()
		self.lmbda = 100.

		if load: self.load(load_pretrain)
		else:
			self.logger = Logger(name)

			self.generator = Unet().to(self.device)
			self.discriminator = Discriminator().to(self.device)

			self.generator.apply(weights_init)
			self.discriminator.apply(weights_init)

		self.pre_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_pre)
		self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5,0.999))
		self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5,0.999))


	def paint(self, gray_image: np.ndarray):
		""" Gets grayscale image, returns painted image """

		# Create transforms to resize image to neatest multiple of 8
		og_shape = gray_image.shape
		new_shape = round_shape(og_shape, 8)

		transform_new = transforms.Resize(new_shape,  Image.BICUBIC)
		transform_og = transforms.Resize(og_shape,  Image.BICUBIC)

		gray_image = np.array(transform_new(Image.fromarray(gray_image, "L")))	# resize to new shape
		
		# paint image
		self.generator.eval()
		with torch.no_grad():
			l = gray_to_tensor(gray_image).to(self.device)
			l_bat = torch.reshape(l, (1, l.size(0), l.size(1), l.size(2)))	# convert to "batch"
			ab = self.generator(l_bat)[0]
			lab_tensor = torch.cat((l, ab), dim=0)
		img = tensor_to_image(lab_tensor)

		img = np.array(transform_og(Image.fromarray(img, "RGB")))	# resize to original shape

		return img


	def optimize_discriminator(self, l_batch, ab_batch, fake_ab_batch, threshold=float('-inf')) -> float:
		""" Do a gradient step for discriminator, return loss """

		self.discriminator.set_gradients(True)
		self.discriminator.train()
		self.optimizer_d.zero_grad()

		fake_images = torch.cat((l_batch, fake_ab_batch), dim=1).detach()
		real_images = torch.cat((l_batch, ab_batch), dim=1)

		fake_preds = self.discriminator(fake_images)
		loss_fake = self.gan_criterion(fake_preds, real_data=False)

		real_preds = self.discriminator(real_images)
		loss_real = self.gan_criterion(real_preds, real_data=True)

		loss = loss_fake + loss_real

		with torch.no_grad():
			avg_accuracy = (torch.sum(1-fake_preds) + torch.sum(real_preds)) / (len(l_batch)*2)

		# if loss.item() > threshold:
		if avg_accuracy < threshold:
			loss.backward()
			self.optimizer_d.step()

		return (loss_fake.item(), loss_real.item(), avg_accuracy.item())
	
	def optimize_generator(self, l_batch, ab_batch, fake_ab_batch) -> float:
		""" Do a gradient step for generator, return loss """

		self.discriminator.set_gradients(False)
		for p in self.discriminator.parameters(): assert not p.requires_grad	# ------------------------
		self.discriminator.eval()
		self.optimizer_g.zero_grad()

		fake_images = torch.cat((l_batch, fake_ab_batch), dim=1)
		fake_preds = self.discriminator(fake_images)
		loss_gan = self.gan_criterion(fake_preds, real_data=True)
		loss_l1 = self.l1_criterion(fake_ab_batch, ab_batch) * self.lmbda
		
		loss = loss_gan + loss_l1

		loss.backward()
		self.optimizer_g.step()

		return (loss_gan.item(), loss_l1.item())
	

	def train_model(self, trainloader) -> tuple[float,float]:
		""" Trains model for 1 epoch """

		self.generator.train()
		loss_gatherer = LossGatherer()

		for l_batch, ab_batch in tqdm(trainloader, desc='Train'):

			# get fake colors
			fake_ab_batch = self.generator(l_batch)

			loss_fake, loss_real, avg_accuracy = self.optimize_discriminator(l_batch, ab_batch, fake_ab_batch, threshold=0.7)
			loss_gan, loss_l1 = self.optimize_generator(l_batch, ab_batch, fake_ab_batch)

			loss_gatherer(loss_fake, loss_real, loss_gan, loss_l1, avg_accuracy)

		self.logger.after_epoch(loss_gatherer, train=True)

	def test_model(self, testloader, pretrain=False, log=False):
		""" Tests model on testset, returns losses """

		self.generator.eval()
		self.discriminator.eval()
		sum_pre_loss = 0.
		loss_gatherer = LossGatherer()

		with torch.no_grad():
			for l_batch, ab_batch in tqdm(testloader, desc='Test'):
				real_images = torch.cat((l_batch, ab_batch), dim=1)
				fake_ab_batch = self.generator(l_batch)
				fake_images = torch.cat((l_batch, fake_ab_batch), dim=1)

				# save some images from batch to logger
				self.logger.add_images(real_images, fake_images)

				# get losses
				loss_l1 = self.l1_criterion(fake_ab_batch, ab_batch).item()
				if pretrain:
					sum_pre_loss += loss_l1
				else:
					fake_preds = self.discriminator(fake_images)
					real_preds = self.discriminator(real_images)

					loss_fake = self.gan_criterion(fake_preds, real_data=False).item()
					loss_real = self.gan_criterion(real_preds, real_data=True).item()
					loss_gan = self.gan_criterion(fake_preds, real_data=True).item()

					loss_gatherer(loss_fake, loss_real, loss_gan, (loss_l1 * self.lmbda))

		if log and not pretrain: self.logger.after_epoch(loss_gatherer, train=False)

		if pretrain: return sum_pre_loss / len(testloader)
		if not log: return loss_gatherer.get_losses()

	def pretrain_generator(self, trainloader) -> None:
		""" Pretrains generator for 1 epoch """

		self.generator.train()
		sum_loss = 0

		for l_batch, ab_batch in tqdm(trainloader, desc='Train'):

			fake_ab_batch = self.generator(l_batch)
			
			loss = self.l1_criterion(fake_ab_batch, ab_batch)
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
		else: self.discriminator.apply(weights_init)


	def count_parameters(self):
		g_pcount = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
		self.discriminator.set_gradients(True)
		d_pcount = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

		return g_pcount, d_pcount

def weights_init(m):
		""" init weights of generator and discriminator """

		classname = m.__class__.__name__
		if hasattr(m, 'weight') and 'Conv' in classname:
			nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
			if hasattr(m, 'bias') and m.bias is not None:
				nn.init.constant_(m.bias.data, 0.0)
		elif 'BatchNorm2d' in classname:
			nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
			nn.init.constant_(m.bias.data, 0.)

def round_shape(tup: tuple[int], c: int) -> tuple[int]:
	""" Used to find nearest multiple of 8 for image size to fit Unet """

	arr = np.array(tup)
	new_shape = (c * np.ceil((2 * arr - c) / (2 * c))).astype(int)
	return tuple(new_shape)
