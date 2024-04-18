import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UnetIn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetIn, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.conv(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upConv(x1)
        x = torch.cat([x2, x1], dim=1)  # dim 1 is channels
        return self.conv(x)

class UnetOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.tanh(self.conv(x))
        

class Unet(nn.Module):
    def __init__(self, networkIn=1, output=2) -> None:
        super(Unet, self).__init__()
        filters = 32
        self.convIn = UnetIn(networkIn, filters)
        
        self.convDown1 = UnetDown(filters, filters*2)
        self.convDown2 = UnetDown(filters*2, filters*4)
        self.convDown3 = UnetDown(filters*4, filters*8)
        self.convDown4 = UnetDown(filters*8, filters*16)
        
        self.convUp4 = UnetUp(filters*16, filters*8)
        self.convUp3 = UnetUp(filters*8, filters*4)
        self.convUp2 = UnetUp(filters*4, filters*2)
        self.convUp1 = UnetUp(filters*2, filters)
        
        self.convOut = UnetOut(filters, output)
        
    def forward(self, x):
        x1 = self.convIn(x)
        
        x2 = self.convDown1(x1)
        x3 = self.convDown2(x2)
        x4 = self.convDown3(x3)
        x5 = self.convDown4(x4)
        
        x = self.convUp4(x5, x4)
        x = self.convUp3(x, x3)
        x = self.convUp2(x, x2)
        x = self.convUp1(x, x1)
        
        x = self.convOut(x)
        return x

