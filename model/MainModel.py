import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.notebook import tqdm

import os
from shutil import copyfile

from network import Unet, Discriminator, GANLoss
from logger import Logger


class Painter(nn.Module):
    def __init__(self, hyparams=None, g_params=None, d_params=None, load_path=None):
        self.logger = Logger(hyparams, g_params, d_params)

        self.generator = Unet(g_params)
        self.discriminator = Discriminator(d_params)

        self.gan_criterion = GANLoss()
        self.l1_criterion = nn.L1Loss()

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=hyparams.lr_g)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=hyparams.lr_d)

    def train_generator(self, trainloader):
        self.generator.train()
        self.discriminator.eval()
        for l_bat, ab_bat in tqdm(trainloader):	# Note: tdqm is a wrapper that shows progress (didnt try it yet)
            # set data to device
            l_bat = l_bat.to(self.device)
            ab_bat = ab_bat.to(self.device)
            
            self.optimizer_g.zero_grad()
            
            fake_ab_bat = self.generator(l_bat)
            fake_image_bat = torch.cat(l_bat, fake_ab_bat, dim=0)
            
            with torch.no_grad():
                preds = self.discriminator(fake_image_bat)
            
            loss_g = self.gan_criterion(preds, real_data = True)
            loss_g.backward()
            

    def train_discriminator(self, trainloader):
        self.generator.eval()
        self.discriminator.train()
        for l_bat, ab_bat in tqdm(trainloader):	# Note: tdqm is a wrapper that shows progress (didnt try it yet)
            # set data to device
            l_bat = l_bat.to(self.device)
            ab_bat = ab_bat.to(self.device)
            
            real_image_bat = torch.cat(l_bat, ab_bat, dim=0)
            
            self.optimizer_g.zero_grad()
            with torch.no_grad():
                fake_ab_bat = self.generator(l_bat)
                fake_image_bat = torch.cat(l_bat, fake_ab_bat, dim=0)
            
            preds = self.discriminator(fake_image_bat)
            loss_d = self.gan_criterion(preds, real_data = False)
            loss_d.backward()
            
            preds = self.discriminator(real_image_bat)
            loss_d = self.gan_criterion(preds, real_data = True)
            loss_d.backward()
            


    def train_model(self, trainloader):
        self.generator.train()
        self.discriminator.train()

        for l_bat, ab_bat in tqdm(trainloader):	# Note: tdqm is a wrapper that shows progress (didnt try it yet)

            # set data to device
            l_bat = l_bat.to(self.device)
            ab_bat = ab_bat.to(self.device)

            real_image_bat = torch.cat(l_bat, ab_bat, dim=0)

            self.optimizer_g.zero_grad()

            # forward
            # ?????????
            fake_ab_bat = self.generator(l_bat)
            fake_image_bat = torch.cat(l_bat, fake_ab_bat, dim=0)
            preds = self.discriminator(fake_image_bat)
            loss_d = self.gan_criterion(preds, real_data=False)
            # backwards
            # ????????

            # log accuracy
            # ??????

            # if (self.global_batch % 30) == 0:
            # 	print(f"batch {self.global_batch:>4d}, loss = {loss.item():.4f}")


    def backward_generator(self):
        return None

    def test_model(self, testloader, prints=True):
        pass


    def save_model(self, backup=True):
        pass

    def load_model(self, path):
        pass

    def count_parameters(self):
        # maybe does not work
        return sum(p.numel() for p in self.gan() if p.requires_grad)

