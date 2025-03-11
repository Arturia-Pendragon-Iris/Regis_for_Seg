from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch.distributions.normal import Normal
from registration_2025 import layers
from torchvision.transforms import transforms
import numpy as np


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, n1=16):
        super(Encoder, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        return (e5, e4, e3, e2, e1)


class Decoder_together(nn.Module):
    def __init__(self, out_ch=2, n1=16):
        super(Decoder_together, self).__init__()

        filters = [n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, embed_1, embed_2):
        e5 = torch.concat((embed_1[0], embed_2[0]), dim=1)
        e4 = torch.concat((embed_1[1], embed_2[1]), dim=1)
        e3 = torch.concat((embed_1[2], embed_2[2]), dim=1)
        e2 = torch.concat((embed_1[3], embed_2[3]), dim=1)
        e1 = torch.concat((embed_1[4], embed_2[4]), dim=1)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)

        return out


class Decoder(nn.Module):
    def __init__(self, out_ch=2, n1=16):
        super(Decoder, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, embed):
        e5 = embed[0]
        e4 = embed[1]
        e3 = embed[2]
        e2 = embed[3]
        e1 = embed[4]

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)

        return out


class VxmDense(nn.Module):
    def __init__(self,
                 inshape,
                 scale=1,
                 ndims=2,
                 int_steps=7,
                 n_features=16,
                 use_morph=True):
        super().__init__()
        self.training = True
        self.scale = scale

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(ndims, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_morph:
            self.integrate = layers.VecInt(inshape, int_steps)
        else:
            self.integrate = None

        self.encoder = Encoder(in_ch=2, n1=n_features)
        self.decoder = Decoder(n1=n_features, out_ch=ndims)

        self.transformer = layers.SpatialTransformer(inshape)
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.1))

    def forward(self, raw_img, fixed_img, smooth=False):

        embed = self.encoder(torch.concat((raw_img, fixed_img), dim=1))
        pos_flow = self.decoder(embed)

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

        if smooth:
            mu = np.random.uniform(low=0.5, high=1)
            pos_flow = self.blur(pos_flow[0]) * mu
            pos_flow = pos_flow.unsqueeze(dim=0)

        # warp image with flow field
        src_re = self.transformer(raw_img, pos_flow)
        return src_re, pos_flow


class VxmDense_together(nn.Module):
    def __init__(self,
                 inshape,
                 scale=1,
                 ndims=2,
                 int_steps=7,
                 n_features=16,
                 use_morph=True):
        super().__init__()
        self.training = True
        self.scale = scale

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(ndims, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_morph:
            self.integrate = layers.VecInt(inshape, int_steps)
        else:
            self.integrate = None

        self.encoder = Encoder(n1=n_features)
        self.decoder = Decoder(n1=n_features, out_ch=ndims)

        self.transformer = layers.SpatialTransformer(inshape)
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 0.5))

    def forward(self, raw_img, fixed_img, smooth=False):

        embed_1 = self.encoder(raw_img)
        embed_2 = self.encoder(fixed_img)

        pos_flow = self.decoder(embed_1, embed_2)

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

        if smooth:
            mu = np.random.uniform(low=0, high=1)
            pos_flow = self.blur(pos_flow[0]) * mu
            pos_flow = pos_flow.unsqueeze(dim=0)

        # warp image with flow field
        src_re = self.transformer(raw_img, pos_flow)
        return src_re, pos_flow
