import torch
import numpy as np

from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop


class ConvolutionalBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(num_features=kwargs["out_channels"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder1 = nn.Sequential(
            ConvolutionalBlock(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        self.encoder2 = nn.Sequential(
            ConvolutionalBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )

        self.encoder3 = nn.Sequential(
            ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.encoder4 = nn.Sequential(
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        )

        self.encoder5 = nn.Sequential(
            ConvolutionalBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.decoder1 = nn.Sequential(
            ConvolutionalBlock(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        )
        self.decoder2 = nn.Sequential(
            ConvolutionalBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.decoder3 = nn.Sequential(
            ConvolutionalBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )

        self.decoder4 = nn.Sequential(
            ConvolutionalBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            ConvolutionalBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1),

        )

        self.crop1 = CenterCrop(size=(64, 64))
        self.crop2 = CenterCrop(size=(128, 128))
        self.crop3 = CenterCrop(size=(256, 256))
        self.crop4 = CenterCrop(size=(512, 512))

        self.out = ConvolutionalBlock(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
        en1 = self.encoder1(x)
        en2 = self.encoder2(self.max_pool(en1))
        en3 = self.encoder3(self.max_pool(en2))
        en4 = self.encoder4(self.max_pool(en3))
        en5 = self.encoder5(self.max_pool(en4))

        de1 = self.up1(en5)

        de1 = torch.cat([self.crop1(en4), de1], dim=1)
        de1 = self.decoder1(de1)

        de2 = self.up2(de1)
        de2 = torch.cat([self.crop2(en3), de2], dim=1)
        de2 = self.decoder2(de2)

        de3 = self.up3(de2)
        de3 = torch.cat([self.crop3(en2), de3], dim=1)
        de3 = self.decoder3(de3)

        de4 = self.up4(de3)
        de4 = torch.cat([self.crop4(en1), de4], dim=1)
        de4 = self.decoder4(de4)

        output = self.out(de4)

        return output
