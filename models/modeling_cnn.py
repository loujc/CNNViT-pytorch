# created by Nikolas_loujc

"""CNN helps ViT See Better"""
import math

from os.path import join as pjoin

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class EarlyConvViT(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.model = 'cnn'
        # n_filter_list = (channels, 64, 128, 128, 256, 256, 512)
        n_filter_list = (channels, 64, 128, 128, 256)
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])
        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=768, # dim设置成1024
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
        self.empty = nn.Identity()
    def forward(self, x):
        # print("input",x.size())
        y = self.conv_layers(x)
        # print("after conv",y.size())
        z = self.empty(y) # 用于链接 require_grad = False 的部分
        # print("after empty",z.size())
        # b, n, _ = x.shape
        return z
