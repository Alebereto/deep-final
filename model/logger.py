import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from data.ImagesDataset import tensor_to_image, gray_tensor_to_image

SAVE_PATH = "results"

class Logger():
	""" Class containing information of model, such as layer dimensions
	 	or training stats. Also has functions for plots and debugging. """

	def __init__(self, name, hyparams) -> None:
		self.name = name
		self.hyparams = hyparams

		self.pretrain_loss = list()	# (train, test) tuples

		self.train_loss = list()	# (generator, discriminator) tuples
		self.test_loss = list()		# (generator, discriminator) tuples

		self.recent_images = deque(maxlen=5) # tuples of (gray, colored, fake) as tensors
		
		self.epochs_pretrained = 0
		self.epochs_trained = 0

	def after_epoch(self, train_loss, test_loss) -> None:
		""" Update values after epoch """

		self.epochs_trained += 1

		self.train_loss.append(train_loss)
		self.test_loss.append(test_loss)

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
			gray = real_images[i][0]
			self.recent_images.append((gray, real_images[i], fake_images[i]))

	def plot_performence(self, pretrain=False, show=True) -> None:

		if pretrain: x = np.arange(1, len(self.pretrain_loss) + 1)
		else: x = np.arange(1, len(self.train_loss) + 1)

		if pretrain:
			g_train_loss, g_test_loss = zip(*self.pretrain_loss)
		else:
			g_train_loss, d_train_loss = zip(*self.train_loss)
			g_test_loss, d_test_loss = zip(*self.test_loss)

			plt.subplot(121)

		plt.plot(x, g_train_loss, color='c', label='Train loss')
		plt.plot(x, g_test_loss, color='g', label='Test loss')
		plt.xlabel('Epoch')
		plt.ylabel('Average Loss')
		plt.title('Generator Loss')
		plt.legend()

		if not pretrain:
			plt.subplot(122)
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
			plt.clf()

	def plot_coloring(self, show=True) -> None:

		img_count = len(self.recent_images)
		if img_count == 0: return

		for i in range(img_count):
			gray, real, fake = self.recent_images[i]

			plt.subplot(img_count,3,(i*3)+1)
			if i == 0: plt.title("Grayscale")
			plt.axis('off')
			plt.imshow(gray_tensor_to_image(gray), cmap='gray')

			plt.subplot(img_count,3,(i*3)+2)
			if i == 0: plt.title("Real Color")
			plt.axis('off')
			plt.imshow(tensor_to_image(real))

			plt.subplot(img_count,3,(i*3)+3)
			if i == 0: plt.title("Fake Color")
			plt.axis('off')
			plt.imshow(tensor_to_image(fake))

		if show: plt.show()
		else:
			model_path = os.path.join(SAVE_PATH, self.name)
			plt.savefig(os.path.join(model_path, 'coloring.png'))
			plt.clf()

