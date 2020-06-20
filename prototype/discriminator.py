import copy
import random
import time
import pdb
import os
import sys

import torch
import numpy as np
import torchvision
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, tanh=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.tanh = tanh

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.tanh:
            out = nn.Tanh()(out)
        else:
            out = self.relu(out)

        return out


class LargeBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, tanh=False):
        super(LargeBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn4 = norm_layer(planes)
        self.conv4 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride
        self.tanh = tanh

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.bn4(out)
        out = self.lrelu(out)

        out = self.conv4(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.tanh:
            out = nn.Tanh()(out)
        else:
            out = self.lrelu(out)

        return out


class Discriminator(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, tanh=False, l2_norm=False):
        super(Discriminator, self).__init__()
        self.l2_norm = l2_norm
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 256
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(512 * block.expansion, 10)
        self.fc_adv = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, tanh=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer, tanh=tanh))

        return nn.Sequential(*layers)

    def _get_mean_covmat(self, feats_norm):
        # TODO: Learn how to calculate this
        means = torch.mean(feats_norm, dim=0)
        print('means', means.size())

        # covmats = np.cov(feats_norm.cpu().detach().numpy(), rowvar=False)
        # print('cov_mat', covmats.shape)
        means = -1
        covmats = -1

        return means, covmats

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.layer4(x)
        feats = self.avgpool(x)
        feats = torch.flatten(feats, 1)

        # TODO: Forget about normalization for now
        # batchsizex512
        # print('feats', feats.size())
        if self.l2_norm:
            feats = F.normalize(feats, p=2, dim=1)
        # batchsizex512
        # print('feats_norm', feats_norm.size())
        # means, covmats = self._get_mean_covmat(feats_norm)

        logits_cls = self.fc_cls(feats)

        logits_adv = self.fc_adv(feats)

        return feats, logits_cls, logits_adv

    def forward(self, x):
        return self._forward_impl(x)


class Discriminator_no_resnet(nn.Module):
    def __init__(self, width=32, l2_norm=False):
        super(Discriminator_no_resnet, self).__init__()
        ndf = 32
        self.l2_norm = l2_norm
        self.fc_cls = nn.Linear(512, 10)
        self.fc_adv = nn.Linear(512, 1)
        self.main = nn.Sequential(
            # state size. (ndf) x 14 x 14
            nn.Conv2d(256, ndf * 2, 3, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, ndf * 16, 2, 1, 0, bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        feats = self.main(input)
        feats = torch.flatten(feats, 1)

        if self.l2_norm:
            feats = F.normalize(feats, p=2, dim=1)

        logits_cls = self.fc_cls(feats)
        logits_adv = self.fc_adv(feats)
        return feats, logits_cls, logits_adv


def discriminator(type, l2_norm=False):
    if type == 'resnet':
        return Discriminator(BasicBlock, [2, 2, 2, 2], tanh=False, l2_norm=l2_norm)
    elif type == 'resnet_large':
        return Discriminator(LargeBlock, [2, 2, 2, 2], tanh=False, l2_norm=l2_norm)
    elif type == 'no_resnet':
        return Discriminator_no_resnet(l2_norm=l2_norm)
    else:
        raise NotImplementedError('Discriminator type not found.')
