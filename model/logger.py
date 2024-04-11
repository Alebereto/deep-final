import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from data.ImagesDataset import tensor_to_image, gray_tensor_to_image

SAVE_PATH = "results"

class Logger():
	""" Class containing information of model, such as layer dimensions
	 	or training stats. Also has functions for plots and debugging. """

	def __init__(self, name, hyparams, g_params, d_params) -> None:
		self.name = name
		self.hyparams = hyparams
		self.g_params = g_params
		self.d_params = d_params

		self.g_test_loss, self.g_train_loss = list(), list()
		self.d_test_loss, self.d_train_loss = list(), list()

		self.recent_images = deque(maxlen=5) # tuples of (gray, colored, fake) as tensors
		
		self.epochs_trained = 0

	def after_epoch(self, train_loss, test_loss) -> None:
		""" Update values after epoch """

		self.epochs_trained += 1

		self.g_train_loss.append(train_loss[0])
		self.d_train_loss.append(train_loss[1])
		self.g_test_loss.append(test_loss[0])
		self.d_test_loss.append(test_loss[1])

	def add_images(self, real_images, fake_images) -> None:
		for i in range(len(real_images)):
			gray = real_images[i][0]
			self.recent_images.append((gray, real_images[i], fake_images[i]))

	def plot_performence(self, show=True) -> None:
		x = np.arange(1, len(self.train_acc) + 1)

		plt.subplot(121)
		plt.plot(x, self.g_train_loss, color='c', label='Train loss')
		plt.plot(x, self.g_test_loss, color='g', label='Test loss')
		plt.xlabel('Epoch')
		plt.ylabel('Average Loss')
		plt.title('Generator Loss')
		plt.legend()

		plt.subplot(122)
		plt.plot(x, self.d_train_loss, color='c', label='Train loss')
		plt.plot(x, self.d_test_loss, color='g', label='Test loss')
		plt.xlabel('Epoch')
		plt.ylabel('Average Loss')
		plt.title('Discriminator Loss')
		plt.legend()

		if show: plt.show()
		else:
			model_path = os.path.join(SAVE_PATH, self.name)
			plt.savefig(os.path.join(model_path, 'performence.png'))
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

