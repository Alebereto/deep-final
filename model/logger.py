import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from data.ImagesDataset import tensor_to_image
import cv2

SAVE_PATH = "results"

class LossGatherer():
	def __init__(self, batch_count) -> None:
		""" Used to gather losses during training and testing epochs """

		self.batch_count = batch_count
		
		self.sum_loss_fake, self.sum_loss_real = 0., 0.		# discriminator losses
		self.sum_gan_loss, self.sum_l1_loss = 0., 0.		# generator losses

	def __call__(self, loss_fake, loss_real, loss_gan, loss_l1) -> None:
		self.sum_loss_fake += loss_fake
		self.sum_loss_real += loss_real
		self.sum_gan_loss += loss_gan
		self.sum_l1_loss += loss_l1

	def get_losses(self):
		loss_fake = self.sum_loss_fake / self.batch_count
		loss_real = self.sum_loss_real / self.batch_count
		gan_loss = self.sum_gan_loss / self.batch_count
		l1_loss = self.sum_l1_loss / self.batch_count

		return (loss_fake, loss_real, gan_loss, l1_loss)
	

class Logger():
	""" Class containing information of model, such as layer dimensions
	 	or training stats. Also has functions for plots and debugging. """

	def __init__(self, name) -> None:
		self.name = name

		self.pretrain_loss = list()	# (train, test) tuples

		# self.train_loss = list()	# (generator, discriminator) tuples # ===============================================================
		# self.test_loss = list()		# (generator, discriminator) tuples

		self.recent_images = deque(maxlen=5) # tuples of (colored, fake) as tensors
		
		self.epochs_pretrained = 0
		self.epochs_trained = 0

	def after_epoch(self, loss_gatherer:LossGatherer, train:bool) -> None:
		""" Update values after epoch """

		self.epochs_trained += 1

		loss_fake, loss_real, gan_loss, l1_loss = loss_gatherer.get_losses()
		if train:
			# add to train losses # ===============================================================
			pass
		else:
			# add to test losses
			pass

	def after_pretrain(self, train_loss, test_loss) -> None:
		""" Update values after pretrain epoch """

		self.epochs_pretrained += 1

		self.pretrain_loss.append((train_loss, test_loss))

	def add_images(self, real_images, fake_images) -> None:
		""" gets batch of real images and fake images from test, saves some of them """

		# get some images from batch
		indeces = np.random.choice(len(real_images), size=5, replace=False)
		real_images, fake_images = real_images[indeces], fake_images[indeces]

		for i in range(len(real_images)):
			self.recent_images.append((real_images[i], fake_images[i]))

	def plot_performence(self, pretrain=False, show=True) -> None:

		if pretrain: x = np.arange(1, len(self.pretrain_loss) + 1)
		else: x = np.arange(1, len(self.train_loss) + 1)

		if pretrain:
			g_train_loss, g_test_loss = zip(*self.pretrain_loss)
		else:
			g_train_loss, d_train_loss = zip(*self.train_loss)
			g_test_loss, d_test_loss = zip(*self.test_loss)

			plt.subplot(211)

		plt.plot(x, g_train_loss, color='c', label='Train loss')
		plt.plot(x, g_test_loss, color='g', label='Test loss')
		plt.xlabel('Epoch')
		plt.ylabel('Average Loss')
		plt.title('Generator Loss')
		plt.legend()

		if not pretrain:
			plt.subplot(212)
			plt.plot(x, d_train_loss, color='c', label='Train loss')
			plt.plot(x, d_test_loss, color='g', label='Test loss')
			plt.xlabel('Epoch')
			plt.ylabel('Average Loss')
			plt.title('Discriminator Loss')
			plt.legend()

		if show: plt.show()
		else:
			im_name = 'pretrain_preformence.png' if pretrain else 'model_preformence.png'
			model_path = os.path.join(SAVE_PATH, self.name)
			plt.savefig(os.path.join(model_path, im_name))
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
		print('A')	# ===============================================================

