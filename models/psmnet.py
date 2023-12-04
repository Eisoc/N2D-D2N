"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


from .common import *

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = convbn(inplanes, planes, 3, stride, pad, dilation) 

        self.conv2 = nn.Sequential(
                        nn.Conv2d(planes, planes, 3, 1, pad, dilation),
                        make_bn_layer(planes)
                        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class PSMEncoder(nn.Module):
    def __init__(self, with_ppm=False):
        super(PSMEncoder, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
                            convbn(3,  32, 3, 2, 1, 1),
                            convbn(32, 32, 3, 1, 1, 1),
                            convbn(32, 32, 3, 1, 1, 1)
                            )

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 2,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 3, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 2,1,1 )

        if with_ppm:
            self.ppm = PPM(
                [32, 32, 64, 128, 128],
                ppm_last_conv_planes=128,
                ppm_inter_conv_planes=128,
                )
        else:
            self.ppm = None

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                make_bn_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation ))

        return nn.Sequential(*layers)

    def forward(self, x):
        output1      = self.firstconv(x)
        output2      = self.layer1(output1)
        output3      = self.layer2(output2)
        output4      = self.layer3(output3)
        output5      = self.layer4(output4)

        if self.ppm is not None:
            output5_2 = self.ppm(output5)
        else:
            output5_2 = None

        return [output1, output2, output3, output4, output5]

if __name__ == '__main__':
    pass
