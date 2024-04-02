import matplotlib.pyplot as plt
import numpy as np

class Logger():
	""" Class containing information of model, such as layer dimensions
	 	or training stats. Also has functions for plots and debugging. """

	def __init__(self, g_params, d_params) -> None:
		self.g_params = g_params
		self.d_params = d_params

		self.train_loss = []
		
		self.epochs_trained = 0

	def after_batch(self, something):
		""" Update values after batch """

		# maybe loss or something
		pass

	def after_epoch(self):
		""" Update values after epoch """

		self.epochs_trained += 1

	def plot_performence(self):
		# copy paste from homework 2
		fig, axs = plt.subplots(1, 2, figsize=(10, 5))
		train_acc, train_loss = zip(*self.train_acc_loss)
		test_acc, test_loss = zip(*self.test_acc_loss)
		x = np.arange(1, len(train_acc) + 1)

		# # plotting accuracy data
		axs[0].plot(x, train_acc, label='train accuracy')
		axs[0].plot(x, test_acc, label='test accuracy')
		# # Adding labels and legend
		axs[0].set_xlabel('Global training batches performed')
		axs[0].set_ylabel('Accuracy')
		axs[0].set_title('Accuracy per batch')
		axs[0].set_ylim([0,1])
		axs[0].legend()

		# # plotting loss data
		axs[1].plot(x, train_loss, label='train loss')
		axs[1].plot(x, test_loss, label='test loss')
		# # Adding labels and legend
		axs[1].set_xlabel('Global training batches performed')
		axs[1].set_ylabel('Loss')
		axs[1].set_title('Loss per batch')
		axs[1].legend()

		# Adjust layout
		plt.tight_layout()
		# Display the plot
		plt.show()

