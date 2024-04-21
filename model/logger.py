import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from data.ImagesDataset import tensor_to_image
import cv2

SAVE_PATH = "results"

class LossGatherer():
	def __init__(self) -> None:
		""" Used to gather losses during training and testing epochs """

		self.count_d = 0
		self.count_g = 0
		
		self.sum_loss_fake, self.sum_loss_real = 0., 0.		# discriminator losses
		self.sum_avg_accuracy = 0.
		self.sum_gan_loss, self.sum_l1_loss = 0., 0.		# generator losses

		self.sum_loss_d, self.sum_loss_g = 0., 0.

	def __call__(self, loss_fake, loss_real, loss_gan, loss_l1, avg_accuracy=None) -> None:
		if loss_fake is not None and loss_real is not None:
			self.sum_loss_fake += loss_fake
			self.sum_loss_real += loss_real

			self.sum_loss_d += loss_fake + loss_real
			self.count_d += 1
			if avg_accuracy is not None: self.sum_avg_accuracy += avg_accuracy

		if loss_fake is not None and loss_real is not None:
			self.sum_gan_loss += loss_gan
			self.sum_l1_loss += loss_l1

			self.sum_loss_g += loss_gan + loss_l1
			self.count_g += 1

	def get_losses(self):
		loss_fake = self.sum_loss_fake / self.count_d
		loss_real = self.sum_loss_real / self.count_d
		avg_accuracy = self.sum_avg_accuracy / self.count_d
		gan_loss = self.sum_gan_loss / self.count_g
		l1_loss = self.sum_l1_loss / self.count_g

		loss_d = self.sum_loss_d / self.count_d
		loss_g = self.sum_loss_g / self.count_g

		if self.sum_avg_accuracy != 0.: return (loss_fake, loss_real, gan_loss, l1_loss, loss_d, loss_g, avg_accuracy)
		return (loss_fake, loss_real, gan_loss, l1_loss, loss_d, loss_g)
	

class Logger():
	""" Class containing information of model, such as layer dimensions
	 	or training stats. Also has functions for plots and debugging. """

	def __init__(self, name) -> None:
		self.name = name

		self.pretrain_loss = list()	# (train, test) tuples

		self.train_losses = list()	# (loss_fake, loss_real, gan_loss, l1_loss, loss_d, loss_g, avg_accuracy) tuples
		self.test_losses = list()	# (loss_fake, loss_real, gan_loss, l1_loss, loss_d, loss_g) tuples

		self.recent_images = deque(maxlen=6) # tuples of (colored, fake) as lab tensors
		
		self.epochs_pretrained = 0
		self.epochs_trained = 0

	def after_epoch(self, loss_gatherer:LossGatherer, train:bool) -> None:
		""" Update values after epoch """

		if train: self.epochs_trained += 1

		losses = loss_gatherer.get_losses()

		if train: self.train_losses.append(losses)
		else: self.test_losses.append(losses)

	def after_pretrain(self, train_loss, test_loss) -> None:
		""" Update values after pretrain epoch """

		self.epochs_pretrained += 1

		self.pretrain_loss.append((train_loss, test_loss))

	def add_images(self, real_images, fake_images) -> None:
		""" gets batch of real images and fake images from test, saves some of them """

		# get some images from batch
		indeces = np.random.choice(len(real_images), size=6, replace=False)
		real_images, fake_images = real_images[indeces], fake_images[indeces]

		for i in range(len(real_images)):
			self.recent_images.append((real_images[i], fake_images[i]))

	def plot_loss(self, train, test=None, title='plot', ylabel='Average Loss', xlabel='Epoch'):

		x = np.arange(1, len(train)+1)

		plt.plot(x, train, color='c', label='Train loss')
		if test is not None: plt.plot(x, test, color='g', label='Test loss')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		if test is not None: plt.legend()

	def plot_performence(self, pretrain=False, show=True) -> None:
		""" Plot losses """

		plots = list()	# list of losses and titles to plot

		if pretrain:
			g_train_loss, g_test_loss = zip(*self.pretrain_loss)
			plots.append((g_train_loss, g_test_loss, 'Pretrain Loss'))
		else:
			(train_loss_fake, train_loss_real, train_gan_loss, train_l1_loss,
				train_loss_d, train_loss_g, avg_accuracy) = zip(*self.train_losses)
			(test_loss_fake, test_loss_real, test_gan_loss, test_l1_loss,
				test_loss_d, test_loss_g) = zip(*self.test_losses)

			plots.append((train_loss_d, test_loss_d, 'Discriminator Loss'))
			plots.append((train_loss_g, test_loss_g, 'Generator Loss'))

			plots.append((train_loss_fake, test_loss_fake, 'Discriminator Fake Loss'))
			plots.append((train_loss_real, test_loss_real, 'Discriminator Real Loss'))
			plots.append((avg_accuracy, None, 'Discriminator Accuracy'))
			plots.append((train_gan_loss, test_gan_loss, 'Generator GAN Loss'))
			plots.append((train_l1_loss, test_l1_loss, 'Generator L1 Loss (times lambda)'))

		model_path = os.path.join(SAVE_PATH, self.name)

		for train, test, title in plots:
			self.plot_loss(train, test, title)

			if show: plt.show()
			else: 	 plt.savefig(f'{model_path}\\{title}.png')
			plt.close()

	def plot_coloring(self, show=True) -> None:

		img_count = len(self.recent_images)
		if img_count == 0: return

		for i in range(img_count):
			real, fake = self.recent_images[i]
			real_img = tensor_to_image(real)
			fake_img = tensor_to_image(fake)
			gray_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)

			plt.subplot(3,img_count,(i+1))
			if i == 0: plt.ylabel("Gray")
			plt.xticks([])
			plt.yticks([])
			plt.imshow(gray_img, vmin=0, vmax=255, cmap='gray')

			plt.subplot(3,img_count,(img_count)+(i+1))
			if i == 0: plt.ylabel("Real")
			plt.xticks([])
			plt.yticks([])
			plt.imshow(real_img, vmin=0, vmax=255)

			plt.subplot(3,img_count,(2*img_count)+(i+1))
			if i == 0: plt.ylabel("Fake")
			plt.xticks([])
			plt.yticks([])
			plt.imshow(fake_img, vmin=0, vmax=255)

		plt.subplots_adjust(hspace=-0.567)
		if show:
			plt.show()
		else:
			model_path = os.path.join(SAVE_PATH, self.name)
			plt.savefig(os.path.join(model_path, 'coloring.png'))
		plt.close()

	def print_epoch(self, idx=-1) -> None:
		train_loss_fake, train_loss_real, train_gan_loss, train_l1_loss, train_loss_d, train_loss_g, _ = self.train_losses[idx]
		test_loss_fake, test_loss_real, test_gan_loss, test_l1_loss, test_loss_d, test_loss_g = self.test_losses[idx]

		print('=====Train Losses=====')
		print(f'Discriminator Loss: {train_loss_d :.6f}, (Fake={train_loss_fake:.6f}), (Real={train_loss_real:.6f})')
		print(f'Generator Loss: {train_loss_g :.6f}, (GAN={train_gan_loss:.6f}), (L1={train_l1_loss:.6f})')

		print('-----Test Losses-----')
		print(f'Discriminator Loss: {test_loss_d :.6f}, (Fake={test_loss_fake:.6f}), (Real={test_loss_real:.6f})')
		print(f'Generator Loss: {test_loss_g :.6f}, (GAN={test_gan_loss:.6f}), (L1={test_l1_loss:.6f})')

