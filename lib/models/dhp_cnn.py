# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

class DHPCNN(nn.Module):
    def __init__(self, in_channels=18):
        super().__init__()
        n_layers = 17
        kernel_size = 3
        strides = [2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dilation = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1]
        pooling = { }#0 : nn.MaxPool2d(2),
                    #3 : nn.MaxPool2d(2) }
        transpose = []#8, 13]
        out_ch = [16, 32, 32, 32, 64, 64, 64, 64, 32, 32, 32, 32, 32, 16, 16, 16, 16]

        self.layers = [nn.Conv2d(in_channels, out_ch[0],
                                 kernel_size,
                                 strides[0],
                                 padding=1,
                                 dilation=dilation[0],
                                 bias=True)]
        
        for l in range(1, n_layers):
            self.layers.append(nn.ReLU(inplace=True))
            if l in transpose:
                self.layers.append(nn.ConvTranspose2d(out_ch[l-1], out_ch[l],
                                                      kernel_size,
                                                      strides[l],
                                                      padding=1,
                                                      output_padding=1,
                                                      bias=False,
                                                      dilation=dilation[l]))
            else:
                self.layers.append(nn.Conv2d(out_ch[l-1], out_ch[l],
                                             kernel_size,
                                             strides[l],
                                             padding=dilation[l],
                                             dilation=dilation[l],
                                             bias=True))
            if l in pooling:
                self.layers.append(pooling[l])
            self.layers.append(nn.BatchNorm2d(out_ch[l]))
                
        self.network = nn.Sequential(*self.layers)
        self.network.apply(self.init_weights)
    def forward(self, x):
        return self.network(x)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, 10.)
            #nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def get_pose_net(config, pretrained, is_train, **kwargs):
    model = DHPCNN()

    return model
