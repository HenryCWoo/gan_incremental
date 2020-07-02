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


class Generator(nn.Module):
    def __init__(self, width=12, deconv=False):
        super(Generator, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        nfms = width
        self.classes_count = 10
        self.vec_size = 10

        self.embed_input = nn.Embedding(self.classes_count, self.vec_size)
        self.embed_lin = nn.Sequential(
            nn.Linear(self.classes_count*self.vec_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        if deconv:
            self.decoder = nn.Sequential(
                # [batch, width*8, 2, 2]
                nn.ConvTranspose2d(512 * 2, nfms * 8, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # [batch, width*4, 4, 4]
                nn.ConvTranspose2d(nfms*8, 256, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # [batch, width*2, 8, 8]
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # [batch, width*1, 16, 16]
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1,
                          padding=0),  # [batch, width*1, 14, 14]
            )
        else:
            self.decoder = nn.Sequential(
                nn.Upsample(size=(2, 2), mode='nearest'),
                nn.Conv2d(512 * 2, nfms*8, kernel_size=2, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size.
                nn.Upsample(size=(4, 4), mode='nearest'),
                nn.Conv2d(nfms*8, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*8) x 4 x 4
                nn.Upsample(size=(8, 8), mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(size=(14, 14), mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            )
        self.output_act = nn.Tanh()

    def forward(self, x):
        # Prepare latent input
        x = x.view(-1, 512, 1, 1)

        # Prepare conditional label input
        y = self.embed_input(y)
        y = y.view(-1, self.classes_count * self.vec_size)  # Unroll embedding
        y = self.embed_lin(y)
        y = y.view(-1, 512, 1, 1)

        cat_input = torch.cat((x, y), dim=1)

        decoded = self.decoder(cat_input)
        decoded = self.output_act(decoded)
        return decoded


class GeneratorBN(nn.Module):
    def __init__(self, width=12, deconv=False):
        super(GeneratorBN, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        nfms = width
        self.classes_count = 10
        self.vec_size = 10

        self.embed_input = nn.Embedding(self.classes_count, self.vec_size)
        self.embed_lin = nn.Sequential(
            nn.Linear(self.classes_count*self.vec_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        if deconv:
            self.decoder = nn.Sequential(
                # [batch, width*8, 2, 2]
                nn.ConvTranspose2d(512 * 2, nfms * 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(nfms * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # [batch, width*4, 4, 4]
                nn.ConvTranspose2d(nfms*8, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # [batch, width*2, 8, 8]
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # [batch, width*1, 16, 16]
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1,
                          padding=0),  # [batch, width*1, 14, 14]
            )
        else:
            self.decoder = nn.Sequential(
                nn.Upsample(size=(2, 2), mode='nearest'),
                nn.Conv2d(512 * 2, nfms * 8, kernel_size=2,
                          stride=1, padding=1),
                nn.BatchNorm2d(nfms * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size.
                nn.Upsample(size=(4, 4), mode='nearest'),
                nn.Conv2d(nfms*8, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*8) x 4 x 4
                nn.Upsample(size=(8, 8), mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(size=(14, 14), mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            )

        self.output_act = nn.Tanh()

    def forward(self, x, y):
        # Prepare latent input
        x = x.view(-1, 512, 1, 1)

        # Prepare conditional label input
        y = self.embed_input(y)
        y = y.view(-1, self.classes_count * self.vec_size)  # Unroll embedding
        y = self.embed_lin(y)
        y = y.view(-1, 512, 1, 1)

        cat_input = torch.cat((x, y), dim=1)

        decoded = self.decoder(cat_input)
        decoded = self.output_act(decoded)
        return decoded


def generator(gen_type, deconv=False):
    if gen_type == 'no_bn':
        return Generator(width=32, deconv=deconv)
    elif gen_type == 'bn':
        return GeneratorBN(width=32, deconv=deconv)
    else:
        raise NotImplementedError("Generator type not recognized.")
