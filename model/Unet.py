import torch
import torch.nn as nn

class UnetIn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetIn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.upConv(x1)
        x = torch.cat([x2, x1], dim=1)  # dim 1 is channels
        return self.conv(x)

class UnetOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        

class Unet(nn.Module):
    def __init__(self, networkIn=1, output=2) -> None:
        super(Unet, self).__init__()
        self.convIn = UnetIn(networkIn, 16)
        
        self.convDown1 = UnetDown(16, 32)
        self.convDown2 = UnetDown(32, 64)
        self.convDown3 = UnetDown(64, 128)
        # self.convDown4 = UnetDown(256, 512)
        
        # self.convUp4 = UnetUp(512, 256)
        self.convUp3 = UnetUp(128, 64)
        self.convUp2 = UnetUp(64, 32)
        self.convUp1 = UnetUp(32, 16)
        
        self.convOut = UnetOut(16, output)
        
    def forward(self, x):
        x1 = self.convIn(x)
        
        x2 = self.convDown1(x1)
        x3 = self.convDown2(x2)
        x4 = self.convDown3(x3)
        # x5 = self.convDown4(x4)
        
        # x = self.convUp4(x5, x4)
        # x = self.convUp3(x, x3)
        x = self.convUp3(x4, x3)
        x = self.convUp2(x, x2)
        x = self.convUp1(x, x1)
        
        x = self.convOut(x)
        return x

