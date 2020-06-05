import copy
import random
import time
import pdb
import os
import sys

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms


# class Generator(nn.Module):
#     def __init__(self, width=12, deconv=False):
#         super(Generator, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         nfms = width
#         if deconv:
#             self.decoder = nn.Sequential(
#                 # [batch, width*8, 2, 2]
#                 nn.ConvTranspose2d(512, nfms * 8, 4, stride=2, padding=1),
#                 nn.BatchNorm2d(nfms * 8),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # [batch, width*4, 4, 4]
#                 nn.ConvTranspose2d(nfms*8, 256, 4, stride=2, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # [batch, width*2, 8, 8]
#                 nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # [batch, width*1, 16, 16]
#                 nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1,
#                           padding=0),  # [batch, width*1, 14, 14]
#             )
#         else:
#             self.decoder = nn.Sequential(
#                 nn.Upsample(size=(2, 2), mode='nearest'),
#                 nn.Conv2d(512, nfms*8, kernel_size=2, stride=1, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size.
#                 nn.Upsample(size=(4, 4), mode='nearest'),
#                 nn.Conv2d(nfms*8, 256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ngf*8) x 4 x 4
#                 nn.Upsample(size=(8, 8), mode='nearest'),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ngf*4) x 8 x 8
#                 nn.Upsample(size=(14, 14), mode='nearest'),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             )
#         self.output_act = nn.Tanh()

#     def forward(self, x):
#         x = x.view(-1, 512, 1, 1)
#         decoded = self.decoder(x)
#         decoded = self.output_act(decoded)
#         return decoded


# def generator(deconv=False):
#     return Generator(width=32, deconv=deconv)


class Generator(nn.Module):
    def __init__(self, width=12, deconv=False):
        super(Generator, self).__init__()

        nfms = width
        self.decoder = nn.Sequential(
            nn.Upsample(size=(2, 2), mode='nearest'),
            nn.Conv2d(2, nfms * 8, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(nfms * 8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size.
            nn.Upsample(size=(4, 4), mode='nearest'),
            nn.Conv2d(nfms * 8, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.Upsample(size=(8, 8), mode='nearest'),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.Upsample(size=(14, 14), mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 2, 16, 16)
        decoded = self.decoder(x)
        return decoded


def generator(deconv=False):
    return Generator()
