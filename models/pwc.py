"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from lib.nn import SynchronizedBatchNorm2d
# from lib.correlation_package.modules.corr import Correlation, Correlation1d
# from sense.lib.correlation import correlation
from lib.correlation_package.correlation import Correlation
from .common import *


class PWCEncoder(nn.Module):
    def __init__(self, with_ppm):
        super(PWCEncoder, self).__init__()
        self.conv1a = convbn(3, 16, kernel_size=3, stride=2)
        self.conv1b = convbn(16, 16, kernel_size=3, stride=1)
        self.conv2a = convbn(16, 32, kernel_size=3, stride=2)
        self.conv2b = convbn(32, 32, kernel_size=3, stride=1)
        self.conv3a = convbn(32, 64, kernel_size=3, stride=2)
        self.conv3b = convbn(64, 64, kernel_size=3, stride=1)
        self.conv4a = convbn(64, 96, kernel_size=3, stride=2)
        self.conv4b = convbn(96, 96, kernel_size=3, stride=1)
        self.conv5a = convbn(96, 128, kernel_size=3, stride=2)
        self.conv5b = convbn(128, 128, kernel_size=3, stride=1)

        if with_ppm:
            self.ppm = PPM([16, 32, 64, 96, 128], ppm_last_conv_planes=128, ppm_inter_conv_planes=128)
        else:
            self.ppm = None

    def forward(self, x):
        c1 = self.conv1b(self.conv1a(x))
        c2 = self.conv2b(self.conv2a(c1))
        c3 = self.conv3b(self.conv3a(c2))
        c4 = self.conv4b(self.conv4a(c3))
        c5 = self.conv5b(self.conv5a(c4))

        if self.ppm is not None:
            c5_2 = self.ppm(c5)
        else:
            c5_2 = None

        return [c1, c2, c3, c4, c5]


# A wrapper on top of the 2D correlation
# We used a C++ implementation in our original code
class Correlation1d(nn.Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation1d, self).__init__()
        self.corr = Correlation(
            pad_size=pad_size,
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride1=stride1,
            stride2=stride2,
            corr_multiply=corr_multiply,
        )
        self.c = max_displacement

    def forward(self, x1, x2):
        cost = self.corr(x1, x2)
        c = self.c
        cost = cost[:, c * (2 * c + 1) : (c + 1) * (2 * c + 1), :, :].contiguous()
        return cost


class PWCFlowDecoder(nn.Module):
    def __init__(
        self,
        md=4,
        encoder_planes=[16, 32, 64, 96, 128],
        refinement_module="none",
        pred_occ=False,
        cat_occ=False,
        upsample_output=False,
    ):
        super(PWCFlowDecoder, self).__init__()
        self.pred_occ = pred_occ
        self.cat_occ = cat_occ
        self.upsample_output = upsample_output

        self.flow_corr = Correlation(
            pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1
        )
        # self.flow_corr = correlation.FunctionCorrelation

        nd = (2 * md + 1) ** 2
        # dd = np.cumsum([128,128,96,64,32]).tolist()
        dd = np.cumsum(encoder_planes).tolist()

        od = nd
        self.flow_conv5_0 = convbn(od, encoder_planes[0], kernel_size=3, stride=1)
        self.flow_conv5_1 = convbn(od + dd[0], encoder_planes[1], kernel_size=3, stride=1)
        self.flow_conv5_2 = convbn(od + dd[1], encoder_planes[2], kernel_size=3, stride=1)
        self.flow_conv5_3 = convbn(od + dd[2], encoder_planes[3], kernel_size=3, stride=1)
        self.flow_conv5_4 = convbn(od + dd[3], encoder_planes[4], kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.flow_deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.flow_upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        if self.pred_occ:
            self.predict_occ5 = predict_class(od + dd[4], 2)

        od = nd + encoder_planes[3] + 4
        if pred_occ and cat_occ:
            od += 2
        self.flow_conv4_0 = convbn(od, encoder_planes[0], kernel_size=3, stride=1)
        self.flow_conv4_1 = convbn(od + dd[0], encoder_planes[1], kernel_size=3, stride=1)
        self.flow_conv4_2 = convbn(od + dd[1], encoder_planes[2], kernel_size=3, stride=1)
        self.flow_conv4_3 = convbn(od + dd[2], encoder_planes[3], kernel_size=3, stride=1)
        self.flow_conv4_4 = convbn(od + dd[3], encoder_planes[4], kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.flow_deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.flow_upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        if self.pred_occ:
            self.predict_occ4 = predict_class(od + dd[4], 2)

        od = nd + encoder_planes[2] + 4
        if pred_occ and cat_occ:
            od += 2
        self.flow_conv3_0 = convbn(od, encoder_planes[0], kernel_size=3, stride=1)
        self.flow_conv3_1 = convbn(od + dd[0], encoder_planes[1], kernel_size=3, stride=1)
        self.flow_conv3_2 = convbn(od + dd[1], encoder_planes[2], kernel_size=3, stride=1)
        self.flow_conv3_3 = convbn(od + dd[2], encoder_planes[3], kernel_size=3, stride=1)
        self.flow_conv3_4 = convbn(od + dd[3], encoder_planes[4], kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.flow_deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.flow_upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        if self.pred_occ:
            self.predict_occ3 = predict_class(od + dd[4], 2)

        od = nd + encoder_planes[1] + 4
        if pred_occ and cat_occ:
            od += 2
        self.flow_conv2_0 = convbn(od, encoder_planes[0], kernel_size=3, stride=1)
        self.flow_conv2_1 = convbn(od + dd[0], encoder_planes[1], kernel_size=3, stride=1)
        self.flow_conv2_2 = convbn(od + dd[1], encoder_planes[2], kernel_size=3, stride=1)
        self.flow_conv2_3 = convbn(od + dd[2], encoder_planes[3], kernel_size=3, stride=1)
        self.flow_conv2_4 = convbn(od + dd[3], encoder_planes[4], kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        if self.pred_occ:
            self.predict_occ2 = predict_class(od + dd[4], 2)

        self.flow_dc_conv1 = convbn(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flow_dc_conv2 = convbn(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.flow_dc_conv3 = convbn(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.flow_dc_conv4 = convbn(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.flow_dc_conv5 = convbn(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.flow_dc_conv6 = convbn(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flow_dc_conv7 = predict_flow(32)
        if pred_occ:
            self.occ_dc_conv7 = predict_class(32, 2)

        in_plane = 2 * encoder_planes[0] + 2
        if refinement_module != "none" and pred_occ and cat_occ:
            in_plane += 2

        self.flow_refinement = make_refinement_module(refinement_module, in_plane, do_flow=True)
        if refinement_module != "none":
            self.predict_occ1 = predict_class(in_plane, 2)

    def num_layers(self):
        return 4 if self.disp_refinement is None else 5

    def forward(self, x1, x2):
        c11, c12, c13, c14, c15 = x1
        c21, c22, c23, c24, c25 = x2

        # flow_warp5 = flow_warp(c25, up_flow6*0.625)
        flow_corr5 = self.flow_corr(c15, c25)
        # flow_corr5 = F.leaky_relu(flow_corr5, negative_slope=0.1)
        flow_corr5 = F.relu(flow_corr5)
        # x = torch.cat((flow_corr5, c15, up_flow6, up_flow_feat6), 1)
        x = torch.cat((flow_corr5, self.flow_conv5_0(flow_corr5)), 1)
        x = torch.cat((x, self.flow_conv5_1(x)), 1)
        x = torch.cat((x, self.flow_conv5_2(x)), 1)
        x = torch.cat((x, self.flow_conv5_3(x)), 1)
        x = torch.cat((x, self.flow_conv5_4(x)), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.flow_deconv5(flow5)
        up_flow_feat5 = self.flow_upfeat5(x)
        if self.pred_occ:
            occ5 = self.predict_occ5(x)

        flow_warp4 = flow_warp(c24, up_flow5 * 1.25)  # 1.25 = 20 / 16
        flow_corr4 = self.flow_corr(c14, flow_warp4)
        # flow_corr4 = F.leaky_relu(flow_corr4, negative_slope=0.1)
        flow_corr4 = F.relu(flow_corr4)
        x = torch.cat((flow_corr4, c14, up_flow5, up_flow_feat5), 1)
        if self.pred_occ and self.cat_occ:
            x = torch.cat((x, F.interpolate(occ5, scale_factor=2, mode="bilinear")), 1)
        x = torch.cat((x, self.flow_conv4_0(x)), 1)
        x = torch.cat((x, self.flow_conv4_1(x)), 1)
        x = torch.cat((x, self.flow_conv4_2(x)), 1)
        x = torch.cat((x, self.flow_conv4_3(x)), 1)
        x = torch.cat((x, self.flow_conv4_4(x)), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.flow_deconv4(flow4)
        up_flow_feat4 = self.flow_upfeat4(x)
        if self.pred_occ:
            occ4 = self.predict_occ4(x)

        flow_warp3 = flow_warp(c23, up_flow4 * 2.5)  # 2.5 = 20 / 8
        flow_corr3 = self.flow_corr(c13, flow_warp3)
        # flow_corr3 = F.leaky_relu(flow_corr3, negative_slope=0.1)
        flow_corr3 = F.relu(flow_corr3)
        x = torch.cat((flow_corr3, c13, up_flow4, up_flow_feat4), 1)
        if self.pred_occ and self.cat_occ:
            x = torch.cat((x, F.interpolate(occ4, scale_factor=2, mode="bilinear")), 1)
        x = torch.cat((x, self.flow_conv3_0(x)), 1)
        x = torch.cat((x, self.flow_conv3_1(x)), 1)
        x = torch.cat((x, self.flow_conv3_2(x)), 1)
        x = torch.cat((x, self.flow_conv3_3(x)), 1)
        x = torch.cat((x, self.flow_conv3_4(x)), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.flow_deconv3(flow3)
        up_flow_feat3 = self.flow_upfeat3(x)
        if self.pred_occ:
            occ3 = self.predict_occ3(x)

        flow_warp2 = flow_warp(c22, up_flow3 * 5.0)  # 5.0 = 20 / 4
        flow_corr2 = self.flow_corr(c12, flow_warp2)
        # flow_corr2 = F.leaky_relu(flow_corr2, negative_slope=0.1)
        flow_corr2 = F.relu(flow_corr2)
        x = torch.cat((flow_corr2, c12, up_flow3, up_flow_feat3), 1)
        if self.pred_occ and self.cat_occ:
            x = torch.cat((x, F.interpolate(occ3, scale_factor=2, mode="bilinear")), 1)
        x = torch.cat((x, self.flow_conv2_0(x)), 1)
        x = torch.cat((x, self.flow_conv2_1(x)), 1)
        x = torch.cat((x, self.flow_conv2_2(x)), 1)
        x = torch.cat((x, self.flow_conv2_3(x)), 1)
        x = torch.cat((x, self.flow_conv2_4(x)), 1)
        flow2 = self.predict_flow2(x)
        if self.pred_occ:
            occ2 = self.predict_occ2(x)

        x = self.flow_dc_conv4(self.flow_dc_conv3(self.flow_dc_conv2(self.flow_dc_conv1(x))))
        x = self.flow_dc_conv6(self.flow_dc_conv5(x))
        flow2_res = self.flow_dc_conv7(x)
        flow2 += flow2_res
        if self.pred_occ:
            occ2_res = self.occ_dc_conv7(x)
            occ2 += occ2_res

        # refinement
        if self.flow_refinement is not None:
            flow1 = F.interpolate(flow2, scale_factor=2, mode="bilinear", align_corners=False)
            flow_warp1 = flow_warp(c21, flow1 * 10.0)
            x = torch.cat((flow1, c11, flow_warp1), 1)
            if self.pred_occ and self.cat_occ:
                x = torch.cat((x, F.interpolate(occ2, scale_factor=2, mode="bilinear")), 1)
            flow1_res, x = self.flow_refinement(x)
            flow1 += flow1_res
            if self.upsample_output:
                flow1 = F.interpolate(flow1, scale_factor=2, mode="bilinear", align_corners=False)
            # flow1 = F.interpolate(flow1, scale_factor=2, mode='bilinear', align_corners=False)
            if self.pred_occ:
                occ1 = F.interpolate(occ2, scale_factor=2, mode="bilinear", align_corners=False)
                occ1_res = self.predict_occ1(x)
                occ1 += occ1_res
                occ1 = F.interpolate(occ1, scale_factor=2, mode="bilinear", align_corners=False)

        if self.upsample_output:
            # flow5 = F.interpolate(flow5, scale_factor=32, mode='bilinear', align_corners=False)
            # flow4 = F.interpolate(flow4, scale_factor=16, mode='bilinear', align_corners=False)
            # flow3 = F.interpolate(flow3, scale_factor=8, mode='bilinear', align_corners=False)
            flow5 = up_flow5
            flow4 = up_flow4
            flow3 = up_flow3
            flow2 = F.interpolate(flow2, scale_factor=4, mode="bilinear", align_corners=False)
            if self.pred_occ:
                occ5 = F.interpolate(occ5, scale_factor=2, mode="bilinear", align_corners=False)
                occ4 = F.interpolate(occ4, scale_factor=2, mode="bilinear", align_corners=False)
                occ3 = F.interpolate(occ3, scale_factor=2, mode="bilinear", align_corners=False)
                occ2 = F.interpolate(occ2, scale_factor=4, mode="bilinear", align_corners=False)

        flow_output = (flow2, flow3, flow4, flow5)
        occ_output = ()
        if self.pred_occ:
            occ_output = (occ2, occ3, occ4, occ5)

        if self.flow_refinement is not None:
            flow_output = (flow1, flow2, flow3, flow4, flow5)
            if self.pred_occ:
                occ_output = (occ1, occ2, occ3, occ4, occ5)

        return (flow_output, occ_output)
