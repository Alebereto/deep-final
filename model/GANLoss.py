import torch
import torch.nn as nn

class GANLoss():
	def __init__(self, device) -> None:
		self.loss_func = nn.BCELoss()
		self.dev = device
	
	def __call__(self, pred, real_data: bool) -> torch.Tensor:
		if real_data: label = torch.ones(pred.size()[0], device=self.dev)
		else: label = torch.zeros(pred.size()[0], device=self.dev)
		loss = self.loss_func(pred, label)
		return loss

