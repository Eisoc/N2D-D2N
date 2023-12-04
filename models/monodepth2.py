"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

from .layers import *


class DepDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.upconv5 = self.build_upconv_layer(128, 128)
        self.upconv4 = self.build_upconv_layer(128, 64)
        self.upconv3 = self.build_upconv_layer(64, 32)
        self.upconv2 = self.build_upconv_layer(32, 32)
        self.upconv1 = self.build_upconv_layer(32, 16)

        self.iconv5 = self.build_iconv_layer(256, 128)
        self.iconv4 = self.build_iconv_layer(128, 64)
        self.iconv3 = self.build_iconv_layer(64, 32)
        self.iconv2 = self.build_iconv_layer(64, 32)
        self.iconv1 = self.build_iconv_layer(16, 16)

        # depth output inform [disparity, probability]
        self.depth_uncer4 = self.build_depth_uncer_layer(64, 2)
        self.depth_uncer3 = self.build_depth_uncer_layer(32, 2)
        self.depth_uncer2 = self.build_depth_uncer_layer(32, 2)
        self.depth_uncer1 = self.build_depth_uncer_layer(16, 2)

        self.upsample5 = nn.Upsample(scale_factor=2)
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

    def build_upconv_layer(self, input_channel, output_channel):
        return nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, 3, padding=1), nn.ELU(inplace=True))

    def build_iconv_layer(self, input_channel, output_channel):
        return nn.Sequential(nn.Conv2d(input_channel, output_channel, 3, padding=1), nn.ELU(inplace=True))

    def build_depth_uncer_layer(self, input_channel, output_channel):
        return nn.Sequential(nn.Conv2d(input_channel, output_channel, 3, padding=1), nn.Sigmoid())

    def forward(self, enc):
        up5_out = self.upconv5(enc[4])
        i5_out = self.iconv5(torch.cat((self.upsample5(up5_out), enc[3]), dim=1))

        up4_out = self.upconv4(i5_out)
        i4_out = self.iconv4(torch.cat((self.upsample5(up4_out), enc[2]), dim=1))
        out4 = self.depth_uncer4(i4_out)

        up3_out = self.upconv3(i4_out)
        i3_out = self.iconv3(torch.cat((self.upsample3(up3_out), enc[1]), dim=1))
        out3 = self.depth_uncer3(i3_out)

        up2_out = self.upconv2(i3_out)
        i2_out = self.iconv2(torch.cat((self.upsample2(up2_out), enc[0]), dim=1))
        out2 = self.depth_uncer2(i2_out)

        up1_out = self.upconv1(i2_out)
        i1_out = self.iconv1(self.upsample1(up1_out))
        out1 = self.depth_uncer1(i1_out)

        return [out1, out2, out3, out4]


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = np.array(num_ch_enc)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return tuple(self.outputs[("disp",i)] for i in range(4))
