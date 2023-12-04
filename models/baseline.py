import torch
import torch.nn as nn
import torch.nn.functional as F

from submodules.submodules import UpSampleBN, norm_normalize

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim
import DispNetS
import FlowNet
import PoseNet
# import DispUnet
import time
import os
from sequence_folders import SequenceFolder
from sequence_folders import testSequenceFolder
from loss_functions import *
from utils_edited import *
from tensorboardX import SummaryWriter
import os
import time
import random
from datetime import datetime
import cv2
import scipy.io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse

crop_size_h = 481
crop_size_w = 641
batch_size = 12

def myfunc_canny(img_ori):
    # img = np.squeeze(img_ori)
    # ori: torch.Size([12, 3, 481, 3])
    img = np.squeeze(img_ori.cpu().numpy())
    # (12, 3, 481, 641)
    # 源代码使用的batch size为1， 所以第一个维度为1,这里会去掉,然后转为numpy数组

    edges_output = np.zeros((batch_size, 1, crop_size_h, crop_size_w), dtype=np.float32)
    # cv2一次只能处理一张图片，预留内存给批处理

    '''
    img = img + 128.0.
    # (12, 3, 481, 641)
    # 可能是8位图像，从[-128, 127]平移到[0, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将3通道的彩色图像转化为1通道的灰度图
    # print(img.shape())
    img = ((img - img.min()) / (img.max() - img.min())) * 255.0
    edges = cv2.Canny(img.astype('uint8'), 100, 220)
    # canny. 100和220是两个阈值
    edges = edges.astype(np.float32)
    edges = edges.reshape((1, crop_size_h, crop_size_w, 1))
    edges = 1 - edges / 255.0
    # 归一化，并且反转，边缘处的像素值接近0，而非边缘处的像素值接近1
    '''

    for i in range(batch_size):
        img = img_ori[i] + 128.0  # Shift values to [0, 255]
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert to [height, width, 3]
        img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        normalized_img = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())) * 255.0
        edge = cv2.Canny(normalized_img.astype('uint8'), 100, 220)
        edges_output[i, 0] = 1 - edge / 255.0

    return torch.from_numpy(edges_output)


def propagate(input_data, dlr, drl, dud, ddu, dim):
    # 传播函数， 实际使用时传入的input data为经过refinement的最终的Depth和Norm
    #  Direction Left to Right, up to down
    if dim > 1:
        dlr = dlr.repeat(1, 1, 1, dim)
        drl = drl.repeat(1, 1, 1, dim)
        dud = dud.repeat(1, 1, 1, dim)
        ddu = ddu.repeat(1, 1, 1, dim)

    # dlr
    x = torch.zeros((1, crop_size_h, 1, dim))
    current_data = torch.cat([x, input_data], dim=2)
    current_data = current_data[:, :, :-1, :]
    # 删去第三维度，W，的最后一列。这样，与x拼接后w和开始一样
    # 因为X为0矩阵，所以拼接完相当于原数据右移一列
    output_data = current_data * dlr + input_data * (1 - dlr)
    # dlr越趋近于1，右移版的权重越大

    # drl 左移
    current_data = torch.cat([output_data, x], dim=2)
    current_data = current_data[:, :, 1:, :]
    output_data = current_data * drl + output_data * (1 - drl)

    # dud 下移
    x = torch.zeros((1, 1, crop_size_w, dim))
    current_data = torch.cat([x, output_data], dim=1)
    current_data = current_data[:, :-1, :, :]
    output_data = current_data * dud + output_data * (1 - dud)

    # ddu 上移
    current_data = torch.cat([output_data, x], dim=1)
    current_data = current_data[:, 1:, :, :]
    output_data = current_data * ddu + output_data * (1 - ddu)

    return output_data


def edges(inputs):
    edge_inputs = myfunc_canny(inputs)
    # print(inputs.shape, edge_inputs.shape, "111111111")
    # torch.Size([12, 3, 481, 641]) torch.Size([12, 1, 481, 641])
    # 边缘处的像素值接近0，而非边缘处的像素值接近1
    edge_inputs = edge_inputs.reshape(12, crop_size_h, crop_size_w, 1)
    # torch.Size([12, 3, 481, 641]) torch.Size([12, 481, 641, 1])
    edge_inputs = edge_inputs.cuda()
    inputs = inputs.permute(0, 2, 3, 1)
    # torch.Size([12, 481, 641, 3]) torch.Size([12, 481, 641, 1])
    # print(inputs.shape, edge_inputs.shape, "222222")
    edge_inputs = torch.cat([edge_inputs, inputs * 0.00784], dim=3)
    # 0.00784=1/127, 注意，除以127的是未经edge处理的inputs，值域还是-128,127

    return edge_inputs


class NNET(nn.Module):
    def __init__(self):
        super(NNET, self).__init__()
        args = None
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=4)

        # args for GEONET
        parser = argparse.ArgumentParser('description: GeoNet')

        parser.add_argument('--is_train', default=1, type=int,
                            help='whether to train or test')

        # Generally fixed parameters
        parser.add_argument('--train_flow', default=False, type=bool,
                            help='whether to train full flow or not')
        parser.add_argument('--sequence_length', default=3, type=int,
                            help='sequence length for each example')
        parser.add_argument('--batch_size', default=4, type=int,
                            help='size of a sample batch')
        parser.add_argument('--epochs', default=30, type=int,
                            help='number of epochs to train on KITTI')
        parser.add_argument('--data_workers', default=8, type=int,
                            help='number of workers')
        parser.add_argument('--img_height', default=128, type=int,
                            help='height of KITTI image')
        parser.add_argument('--img_width', default=416, type=int,
                            help='width of KITTI image')
        parser.add_argument('--num_source', default=2, type=int,
                            help='number of source images')
        parser.add_argument('--num_scales', default=4, type=int,
                            help='number of scaling points')
        parser.add_argument('--seed', default=8964, type=int,
                            help='torch random seed')

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_baseline')
        # Dataset directories
        parser.add_argument('--data_dir', default=os.path.join(base_dir, 'data'),  # /data/kitti_eigen_full/',
                            help='directory of training dataset')
        parser.add_argument('--test_dir', default=os.path.join(base_dir, 'data'),  # '/ceph/data/kitti_raw/',
                            help='directory of testing dataset')

        # To edit during training
        parser.add_argument('--ckpt_dir', default=os.path.join(base_dir, 'models'),
                            help='directory to save checkpoints')

        parser.add_argument('--graphs_dir', default=os.path.join(base_dir, 'graphs'),
                            # default='/ceph/raunaks/GeoNet-PyTorch/reconstruction/graphs/no-bn',
                            help='directory to store tensorboard images and scalars')
        parser.add_argument('--output_ckpt_iter', default=5000, type=int,
                            help='interval to save checkpoints')

        # To edit during evaluation
        parser.add_argument('--outputs_dir', default=os.path.join(base_dir, 'outputs'),
                            # default='/ceph/raunaks/GeoNet-PyTorch/reconstruction/outputs/',
                            help='outer directory to save output depth models')
        parser.add_argument('--ckpt_index', default=150000, type=int,
                            help='the model index to consider while evaluating')

        # Training hyperparameters
        parser.add_argument('--simi_alpha', default=0.85, type=float,
                            help='alpha weight between SSIM and L1 in reconstruction loss')
        parser.add_argument('--loss_weight_rigid_warp', default=1.0, type=float,
                            help='weight for warping by rigid flow')
        parser.add_argument('--loss_weight_disparity_smooth', default=0.5, type=float,
                            help='weight for disp smoothness')
        parser.add_argument('--learning_rate', default=0.0002, type=float,
                            help='learning rate for Adam Optimizer')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum for Adam Optimizer')
        parser.add_argument('--beta', default=0.999, type=float,
                            help='beta for Adam Optimizer')
        parser.add_argument('--weight_decay', default=0, type=float,
                            help='weight decay for Adam Optimizer')

        """
        parser.add_argument('--geometric_consistency_alpha', default=3.0)
        parser.add_argument('--geometric_consistency_beta', default=0.05)
        parser.add_argument('--loss_weight_full_warp', default=1.0)
        parser.add_argument('--loss_weigtht_full_smooth', default=0.2)
        parser.add_argument('--loss_weight_geometrical_consistency', default=0.2)
        """

        args_geonet = parser.parse_args()
        self.geonet = GeoNetModel(args_geonet, device)
        ##

        # for D2N
        self.mean_BGR = [104.008, 116.669, 122.675]  # for ImageNET
        self.crop_size = 320
        self.crop_size_h = 481
        self.crop_size_w = 641
        self.batch_size = 12
        self.k = 9
        self.rate = 4
        self.thresh = 0.95

        self.data_directory = os.path.join(base_dir, 'img_inputs')
        self.inputs = self.input_producer()
        print(self.inputs.shape, "after input producer")
        # torch.Size([12, 3, 481, 641])
        self.inputs = self.bgr_preprocessing(self.inputs)
        print(self.inputs.shape, "after BGR")
        # torch.Size([12, 3, 481, 641]) after BGR

        # D2N refinement
        # Define the Convolution layers
        self.conv1_noise = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_noise2 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_noise = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_noise2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_noise = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_noise2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1_noise = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.encode_norm_noise = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1)

        # Define new convolution layers
        self.conv1_norm_noise_new = nn.Conv2d(3, 128, kernel_size=3, dilation=2, padding=2)
        self.conv2_norm_noise_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv1_norm_noise_new1 = nn.Conv2d(3, 128, kernel_size=3, dilation=2, padding=2)
        self.conv2_norm_noise_new1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm_conv3_noise_new = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

        self.edge_inputs = edges(self.inputs)
        print(self.edge_inputs.shape, "after edge")

        # N2D Refinement
        self.conv1_1 = nn.Conv2d(self.edge_inputs.shape[1], 32, kernel_size=3, dilation=2, padding=2)
        nn.init.xavier_uniform_(self.conv1_1.weight)
        nn.init.constant_(self.conv1_1.bias, 0)

        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2)
        nn.init.xavier_uniform_(self.conv1_2.weight)
        nn.init.constant_(self.conv1_2.bias, 0)

        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2)
        nn.init.xavier_uniform_(self.conv1_3.weight)
        nn.init.constant_(self.conv1_3.bias, 0)

        # Next 3 conv layers without dilation (default is 1)
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2_1.weight)
        nn.init.constant_(self.conv2_1.bias, 0)

        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2_2.weight)
        nn.init.constant_(self.conv2_2.bias, 0)

        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2_3.weight)
        nn.init.constant_(self.conv2_3.bias, 0)

        # Final conv layer
        self.edge_weight = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.edge_weight.weight)
        nn.init.constant_(self.edge_weight.bias, 0)

    def preprocessing(self, img):
        # 调整尺寸
        img = transforms.Resize((self.crop_size_h, self.crop_size_w))(img)

        # 如果随机数小于0.5，则进行水平翻转
        if torch.rand(1).item() < 0.5:
            img = transforms.functional.hflip(img)

        # 修改通道顺序为BGR
        img = transforms.functional.to_tensor(img)
        # img = img[[2, 1, 0], :, :]
        # print(img.shape) 结果torch.Size([3, 481, 641])

        return img

    def input_producer(self):
        composed_transforms = transforms.Compose([
            transforms.Lambda(self.preprocessing)
        ])

        # 使用ImageFolder加载数据
        dataset = ImageFolder(root=self.data_directory, transform=composed_transforms)

        # 使用DataLoader批处理数据
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        batch_images, _ = next(iter(dataloader))  # "_" 无标签
        batch_images = batch_images.to(device)

        return batch_images

    def bgr_preprocessing(self, inputs):
        # inputs :[12, 3, 481, 641]
        if len(inputs.shape) == 3:  # 如果非批量，则增加一个维度
            inputs = inputs.unsqueeze(0)
            print("1", inputs.shape)
        mean_bgr = torch.tensor(self.mean_BGR).reshape(1, 3, 1, 1).to(inputs.device)
        inputs = inputs[:, [2, 1, 0], :, :] + mean_bgr
        # print(mean_bgr.shape, "mean bgr")
        # torch.Size([12, 3, 481, 641]) after BGR
        return inputs

    def forward(self, x, **kwargs):
        out = self.decoder(self.encoder(x), **kwargs)

        # def grid
        x_linspace = torch.linspace(-0.6, 0.6, self.crop_size_w)
        y_linspace = torch.linspace(-0.4, 0.4, self.crop_size_h)
        x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
        z_coordinates = torch.ones_like(x_coordinates)
        grid = torch.stack([x_coordinates, y_coordinates, z_coordinates], dim=-1)
        grid = grid.unsqueeze(0)
        grid = grid.repeat(self.batch_size, 1, 1, 1)

        # Bilinearly upsample the output to match the input resolution
        up_out = F.interpolate(out, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)
        # L2-normalize the first three channels / ensure positive value for concentration parameters (kappa)
        up_out = norm_normalize(up_out)
        pre_norm = up_out
        self.geonet.train()
        pre_depth = self.geonet.test_depth()

        # -----------------------D2N---------------------------------
        fc8_upsample_norm = pre_norm.squeeze()
        fc8_upsample_norm = fc8_upsample_norm.reshape(self.batch_size, self.crop_size_h, self.crop_size_w, 3)

        # Compute norm_matrix similar to tf.extract_image_patches
        # 批处理：
        norm_matrix = F.unfold(fc8_upsample_norm, self.k, dilation=self.rate, stride=1,
                               padding=(self.k + (self.k - 1) * (self.rate - 1) - 1) // 2)
        # tf.extract_image_patches 的输出形状为 [batch, out_height, out_width, k*k*channels]
        # F.unfold的输出是[batch_size, channels*k*k, out_height*out_width]
        # 所以对于我们的unfold，需要转置后两个维度，然后reshape
        norm_matrix = norm_matrix.transpose(1, 2).contiguous()
        # norm_matrix = norm_matrix.reshape([-1, self.crop_size_h, self.crop_size_w,
        # self.k * self.k * fc8_upsample_norm.size(1)])
        # 单张处理：
        # norm_matrix = F.unfold(fc8_upsample_norm.unsqueeze(0), self.k, dilation=self.rate, stride=1, padding=0)
        # .squeeze(0)

        # Compute angle and valid_condition
        matrix_c = norm_matrix.reshape(self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3)

        fc8_upsample_norm = fc8_upsample_norm.unsqueeze(-1)
        # self.batch_size, self.crop_size_h, self.crop_size_w, 3
        # 变为self.batch_size, self.crop_size_h, self.crop_size_w, 3,1

        angle = torch.matmul(matrix_c, fc8_upsample_norm)
        valid_condition = angle > self.thresh
        valid_condition_all = valid_condition.repeat(1, 1, 1, 1, 3)

        # Depth and point processing
        fc8_upsample = pre_depth
        exp_depth = torch.exp(fc8_upsample * 0.69314718056)  # e^0.69 = 2, torch.exp(fc8_upsample*0.69314718056) = 2^fc8
        depth_repeat = exp_depth.repeat(1, 1, 1, 3)
        # (batch_size, h, w, 3). d复制了三次，之后将分别与grid的xyz相乘，将grid的2d转化为3d
        points = grid * depth_repeat  # grid : z = 1, x : [-0.6, 0.6], y : [-0.4, 0.4] uniform distribution

        # Extract point_matrix
        point_matrix = F.unfold(points, self.k, dilation=self.rate, stride=1,
                                padding=(self.k + (self.k - 1) * (self.rate - 1) - 1) // 2)
        point_matrix = point_matrix.transpose(1, 2).contiguous()
        # point_matrix = point_matrix.reshape([-1, self.crop_size_h,
        # self.crop_size_w, self.k * self.k * points.size(1)])

        matrix_a = point_matrix.reshape(self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3)

        # Conditional operations and matrix manipulations
        matrix_a_zero = torch.zeros_like(matrix_a)
        matrix_a_valid = torch.where(valid_condition_all, matrix_a, matrix_a_zero)
        # condition为True时，选择matrix_a中的值，否则选择matrix_a_zero中的值

        matrix_a_trans = matrix_a_valid.permute(0, 1, 2, 4, 3)
        # self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3，然后调换后两个维度
        point_multi = torch.matmul(matrix_a_trans, matrix_a_valid)
        matrix_b = torch.ones(self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 1,
                              dtype=torch.float32)

        matrix_deter = torch.det(point_multi.cpu())  # 行列式，变为三个维度，batch_size, height, width
        inverse_condition = matrix_deter > 1e-5
        inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)  # 重新扩展为5个维度
        inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)

        diag_constant = torch.ones([3])
        diag_element = torch.diag(diag_constant).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # 创建一个形状为 [3, 3] 的对角矩阵，其对角线元素取自diag_constant向量，其他位置为0。
        # 三次调用unsqueeze在张量的开始处添加三个新维度。diag_element的形状从 [3, 3] 变为 [1, 1, 1, 3, 3]。
        diag_matrix = diag_element.repeat(self.batch_size, self.crop_size_h, self.crop_size_w, 1, 1)

        inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)

        # ----------------------- D2N refinement -----------------------
        inv_matrix = torch.inverse(inversible_matrix)

        # Generate Norm
        generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)

        # Normalize
        norm_normalized = F.normalize(generated_norm, p=2, dim=3)  # L2 Norm

        # Scale
        norm_scale = norm_normalized * 10.0
        norm_scale = norm_scale.reshape(-1, self.crop_size_h, self.crop_size_w, 3)  # reshape operation

        # Layers
        x = F.relu(self.conv1_noise(norm_scale))
        x = F.relu(self.conv1_noise2(x))

        x = F.max_pool2d(x, 3, stride=2, padding=1)

        x = F.relu(self.conv2_noise(x))
        x = F.relu(self.conv2_noise2(x))

        x = F.relu(self.conv3_noise(x))
        x = F.relu(self.conv3_noise2(x))

        # FC
        x = F.relu(self.fc1_noise(x))
        encode_norm_upsample_noise = F.interpolate(self.encode_norm_noise(x), size=(self.crop_size_h, self.crop_size_w),
                                                   mode='nearest')

        # Summation and final unit norm
        sum_norm_noise = norm_normalized + encode_norm_upsample_noise
        norm_pred_noise = F.normalize(sum_norm_noise, p=2, dim=3)

        # Concatenate and pass through new layers
        norm_pred_all = torch.cat([fc8_upsample_norm.squeeze(0).unsqueeze(0), norm_pred_noise,
                                   self.inputs * 0.00392156862], dim=3)
        # 0.00392156862 = 1/255. fc8_upsample_norm:VGG生成后，初步处理的深度,已经归一化。norm_pred_noise:D2N,归一化

        norm_pred_all = F.relu(self.conv1_norm_noise_new(norm_pred_all))
        norm_pred_all = F.relu(self.conv1_norm_noise_new1(norm_pred_all))

        norm_pred_all = F.relu(self.conv2_norm_noise_new(norm_pred_all))
        norm_pred_all = F.relu(self.conv2_norm_noise_new1(norm_pred_all))

        norm_pred_final = self.norm_conv3_noise_new(norm_pred_all)

        # Final unit norm
        norm_pred_final = F.normalize(norm_pred_final, p=2, dim=3)

        # ----------------------- N2D -----------------------
        grid_patch = F.unfold(grid, kernel_size=(self.k, self.k), stride=1, dilation=self.rate,
                              padding=(self.k + (self.k - 1) * (self.rate - 1) - 1) // 2)
        grid_patch = grid_patch.transpose(1, 2).contiguous()

        grid_patch = grid_patch.reshape(self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3)

        depth_data = matrix_a[..., 2:3]
        # matrix_a: self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3
        # depth_data获得了matrix_a的最后一个维度中的第三个元素，
        # (self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 1)

        lower_matrix = torch.matmul(matrix_c, grid.unsqueeze(4))
        condition = lower_matrix > 1e-5
        lower = torch.where(condition, 1. / lower_matrix, torch.ones_like(lower_matrix))

        valid_angle = torch.where(condition, angle, torch.zeros_like(angle))
        upper = (matrix_c * grid_patch).sum(dim=-1)
        ratio = (lower * upper.unsqueeze(4))

        estimate_depth = ratio * depth_data

        summed = valid_angle.sum(dim=(3, 4), keepdim=True) + 1e-5
        reciprocal_sum = 1.0 / summed
        reciprocal_sum_repeated = reciprocal_sum.repeat(1, 1, 1, 81, 1)
        valid_angle = valid_angle * reciprocal_sum_repeated

        depth_stage1 = (estimate_depth * valid_angle).sum(dim=[3, 4])
        depth_stage1 = depth_stage1.squeeze().unsqueeze(2).clamp(0, 10.0)
        # clamp确保depth_stage1中的所有值都在[0, 10.0]

        exp_depth = exp_depth.squeeze().unsqueeze(2)

        # Combine depth values and pass through convolution layers
        # 00392156862 = 1/255
        depth_all = torch.cat([depth_stage1, exp_depth, self.inputs.squeeze() * 0.00392156862], dim=2)
        depth_all = depth_all.unsqueeze(0)

        depth_pred_all = F.relu(self.conv1(depth_all))
        depth_pred_all = F.relu(self.conv2(depth_pred_all))
        final_depth = self.conv3(depth_pred_all)

        # ----------------------- N2D Refinement-----------------------

        edges_encoder = self.conv1_1(self.edge_inputs)
        edges_encoder = self.conv1_2(edges_encoder)
        edges_encoder = self.conv1_3(edges_encoder)

        edges_encoder = self.conv2_1(edges_encoder)
        edges_encoder = self.conv2_2(edges_encoder)
        edges_encoder = self.conv2_3(edges_encoder)

        edges_predictor = self.edge_weight(edges_encoder)

        edges_all = edges_predictor + self.edge_inputs.repeat(1, 1, 1, 8)
        edges_all = torch.clamp(edges_all, 0.0, 1.0)

        dlr, drl, dud, ddu, nlr, nrl, nud, ndu = torch.split(edges_all, edges_all.size(3) // 8, dim=3)

        edge_input_depth = final_depth
        edge_input_norm = norm_pred_final

        for _ in range(4):
            final_depth = propagate(edge_input_depth, dlr, drl, dud, ddu, 1)

        for _ in range(4):
            norm_pred_final = propagate(edge_input_norm, nlr, nrl, nud, ndu, 3)
            norm_pred_final = F.normalize(norm_pred_final, dim=3)

        return pre_norm, pre_depth, norm_pred_final, final_depth

    def train(self, mode=True):

        print(self.inputs)

        norm_init, depth_init, depth_final, norm_final = self.forward(self.inputs)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# Decoder (no pixel-wise MLP, no uncertainty-guided sampling)
class Decoder(nn.Module):
    def __init__(self, num_classes=4):
        super(Decoder, self).__init__()
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
        self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
        self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
        self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out


class GeoNetModel(object):
    def __init__(self, args, device):
        self.args = args

        # Nets preparation
        self.disp_net = DispNetS.DispNetS()
        self.pose_net = PoseNet.PoseNet(args.num_source)

        # input channels: src_views * (3 tgt_rgb + 3 src_rgb + 3 warp_rgb + 2 flow_xy +1 error )
        # self.flow_net = FlowNet.FlowNet(12, self.config['flow_scale_factor'])

        if device.type == 'cuda':
            self.disp_net.cuda()
            self.pose_net.cuda()
            # self.flow_net.cuda()

        # Weight initialization
        if (not args.train_flow) and args.is_train:
            print('Initializing weights from scratch')
            # self.disp_net.init_weight()
            # self.pose_net.init_weight()
            self.disp_net.init_weights()
            self.pose_net.init_weight()

        if not args.is_train:

            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

            path = '{}/{}_{}'.format(args.ckpt_dir, 'rigid_depth', str(args.ckpt_index) + '.pth')
            print('Loading saved model weights from {}'.format(path))
            ckpt = torch.load(path)
            self.disp_net.load_state_dict(ckpt['disp_net_state_dict'])
            self.pose_net.load_state_dict(ckpt['pose_net_state_dict'])

        """
        else:
            ckpt = torch.load(config['ckpt_path'])
            self.disp_net.load_state_dict(ckpt['disp_net_state_dict'])
            self.pose_net.load_state_dict(ckpt['pose_net_state_dict'])
            if train_flow:
                if 'flow_net_state_dict' in ckpt:
                    self.flow_net.load_state_dict(ckpt['flow_net_state_dict'])
                else:
                    self.flow_net.init_weight()
        """

        # cudnn.benchmark = True
        # for multiple GPUs
        # self.disp_net = torch.nn.DataParallel(self.disp_net)
        # self.pose_net = torch.nn.DataParallel(self.pose_net)

        self.nets = {
            'disp': self.disp_net,
            'pose': self.pose_net
            # 'flow': self.flow_net
        }

        if args.is_train:
            if not os.path.exists(args.graphs_dir):
                os.makedirs(args.graphs_dir)

            self.tensorboard_writer = SummaryWriter(logdir=args.graphs_dir, flush_secs=30)

            print('Writing graphs to {}'.format(args.graphs_dir))

    def preprocess_test_data(self, sampled_batch):
        """
        sampled_batch: (batch_size, img_height, img_width, channels)
        """
        args = self.args

        tgt_view = sampled_batch
        tgt_view = tgt_view.to(device).float()
        tgt_view *= 1. / 255.
        self.tgt_view = tgt_view * 2.0 - 1.0

        # shape:  #scale, #batch, #chnls, h,w
        self.tgt_view_pyramid = scale_pyramid(self.tgt_view, args.num_scales)
        # shape:  #scale, #batch*#src_views, #chnls,h,w
        self.tgt_view_tile_pyramid = [
            self.tgt_view_pyramid[scale].repeat(args.num_source, 1, 1, 1)
            for scale in range(args.num_scales)
        ]

        self.src_views = None
        self.intrinsics = None
        self.src_views_concat = None
        self.src_views_pyramid = None
        self.multi_scale_intrinsices = None

    def iter_data_preparation(self, sampled_batch):
        args = self.args
        # sampled_batch: tgt_view, src_views, intrinsics

        # shape: batch, ch, h,w
        tgt_view = sampled_batch[0]

        # shape: batch, num_source*ch, h, w
        src_views = sampled_batch[1]

        # shape: batch, 3, 3
        intrinsics = sampled_batch[2]

        # The images here are integral (0-255)
        # shape: batch, 3, h, w
        self.tgt_view = tgt_view.to(device).float()
        self.tgt_view *= 1. / 255.
        self.tgt_view = self.tgt_view * 2. - 1.

        self.src_views = src_views.to(device).float()
        self.src_views *= 1. / 255.
        self.src_views = self.src_views * 2. - 1.
        # print(self.src_views, self.tgt_view)

        self.intrinsics = intrinsics.to(device).float()
        # shape: b*src_views,3,h,w
        self.src_views_concat = torch.cat([
            self.src_views[:, 3 * s:3 * (s + 1), :, :]
            for s in range(args.num_source)
        ], dim=0)

        # shape:  #scale, #batch, h,w, ch
        self.tgt_view_pyramid = scale_pyramid(self.tgt_view, args.num_scales)

        # shape:  #scale, #batch*#src_views, #chnls,h,w
        self.tgt_view_tile_pyramid = [
            self.tgt_view_pyramid[scale].repeat(args.num_source, 1, 1, 1)
            for scale in range(args.num_scales)
        ]

        # shape: scales, b*src_views, h, w, ch
        self.src_views_pyramid = scale_pyramid(self.src_views_concat,
                                               args.num_scales)

        # output multiple disparity prediction
        self.multi_scale_intrinsices = compute_multi_scale_intrinsics(
            self.intrinsics, args.num_scales)

    def spatial_normalize(self, disp):
        curr_c, _, curr_h, curr_w = list(disp.size())
        disp_mean = torch.mean(disp, dim=(0, 2, 3), keepdim=True)
        disp_exp = disp_mean.expand(disp.size())
        return disp / disp_exp

    def build_dispnet(self):
        args = self.args
        # shape: batch, channels, height, width
        self.dispnet_inputs = self.tgt_view

        # for multiple disparity predictions,
        # cat tgt_view and src_views along the batch dimension
        if args.is_train:
            for s in range(args.num_source):  # opt.num_source = 3 - 1 = 2
                self.dispnet_inputs = torch.cat((self.dispnet_inputs, self.src_views[:, 3 * s: 3 * (s + 1), :, :]),
                                                dim=0)
            # [12, 3, 128, 416] - bs*3, channels, height, width

        # shape: pyramid_scales, #batch+#batch*#src_views, h,w
        self.disparities = self.disp_net(self.dispnet_inputs)
        self.loss_disparities = [d.squeeze(1).unsqueeze(3) for d in self.disparities]

        """
        Length = 4
        disparities[0]: (12, 1, 128, 416)
        disparities[1]: (12, 1, 64, 208)
        disparities[2]: (12, 1, 32, 104)
        disparities[3]: (12, 1, 16, 52)
        """
        # shape: pyramid_scales, bs, h,w

        # self.depth = [self.spatial_normalize(disp) for disp in self.disparities]

        self.depth = [1.0 / disp for disp in self.disparities]

        self.depth = [d.squeeze_(1) for d in
                      self.depth]  # is this necessary? Yes, in the tf implementation it is done inside the compute_rigid_flow function

        self.loss_depth = [d.unsqueeze(3) for d in self.depth]

        #         print(self.depth)
        """
        For training data:
        Length = 4
        depth[0]: (12, 128, 416)
        depth[1]: (12, 64, 208)
        depth[2]: (12, 32, 104)
        depth[3]: (12, 16, 52)
        i.e. (batch_size*num_imgs, height, width)
        """

    def build_posenet(self):
        self.posenet_inputs = torch.cat((self.tgt_view, self.src_views), dim=1)
        self.poses = self.pose_net(self.posenet_inputs)
        # (batch_size, num_source, 6)

    def build_rigid_warp_flow(self):
        global n_iter
        # NOTE: this should be a python list,
        # since the sizes of different level of the pyramid are not same
        """
        Uses self.poses and self.depth, computed through build_posenet() and build_dispnet(), respectively
        """
        #         import pickle

        #         infile = open('/ceph/raunaks/depth2.pkl', 'rb')
        #         self.depth = pickle.load(infile)
        #         self.depth = [torch.tensor(d).squeeze(3) for d in self.depth]
        #         print(self.depth[0].size())

        #         infile = open('/ceph/raunaks/pose2.pkl', 'rb')
        #         self.poses = pickle.load(infile)
        #         self.poses = torch.tensor(self.poses)
        #         print(self.poses.shape)

        #         infile = open('/ceph/raunaks/intrin2.pkl', 'rb')
        #         self.multi_scale_intrinsices = torch.tensor(pickle.load(infile))
        #         print(self.multi_scale_intrinsices.shape)

        args = self.args
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []

        # print(self.depth[0].shape)
        for scale in range(args.num_scales):  # num_scales is 4

            for src in range(args.num_source):  # num_source is 2
                # self.depth: (4, 12, _, _)
                # self.poses: (4, 2, 6)
                # self.multi_scale_intrinsices: (4, 4, 3, 3)

                # (4, h, w, 2) for each particular scale
                fwd_rigid_flow = compute_rigid_flow(  # Checks out
                    self.poses[:, src, :],
                    self.depth[scale][:args.batch_size, :, :],  # the first disparity
                    self.multi_scale_intrinsices[:, scale, :, :], False)

                # (4, h, w, 2)
                bwd_rigid_flow = compute_rigid_flow(
                    self.poses[:, src, :],
                    self.depth[scale][args.batch_size * (
                            src + 1):args.batch_size * (src + 2), :, :],
                    self.multi_scale_intrinsices[:, scale, :, :], True)

                if not src:
                    fwd_rigid_flow_cat = fwd_rigid_flow
                    bwd_rigid_flow_cat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_cat = torch.cat(
                        (fwd_rigid_flow_cat, fwd_rigid_flow), dim=0)
                    bwd_rigid_flow_cat = torch.cat(
                        (bwd_rigid_flow_cat, bwd_rigid_flow), dim=0)

            # After the inner loop runs: fwd_rigid_flow_cat - (b*src_imgs, h, w, 2)

            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)

        # After the outer loop runs: fwd_rigid_flow_pyramid: (scales, b*src_imgs, h, w, 2) like (4, 8, h, w, 2)

        self.fwd_rigid_warp_pyramid = [
            flow_warp(self.src_views_pyramid[scale],
                      self.fwd_rigid_flow_pyramid[scale])
            for scale in range(args.num_scales)
        ]

        #         print(self.fwd_rigid_warp_pyramid[0].shape, self.fwd_rigid_warp_pyramid) - different
        #         print(self.tmp_pyramid[0].shape, self.tmp_pyramid)

        self.bwd_rigid_warp_pyramid = [
            flow_warp(self.tgt_view_tile_pyramid[scale],
                      self.bwd_rigid_flow_pyramid[scale])
            for scale in range(args.num_scales)
        ]

        # print(len(self.fwd_rigid_warp_pyramid), " ", self.fwd_rigid_warp_pyramid[0].size())
        # fwd_rigid_warp_pyramid: (8,128,416,3), (8,64,208,3), (8,32,104,3), (8,16,52,3)

        if n_iter % 10000 == 0:
            for j in range(len(self.fwd_rigid_warp_pyramid)):
                x = self.fwd_rigid_warp_pyramid[j].permute(0, 3, 1, 2)
                x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                self.tensorboard_writer.add_images('fwd_rigid_warp_scale' + str(j), x, n_iter)

            for j in range(len(self.bwd_rigid_warp_pyramid)):
                x = self.fwd_rigid_warp_pyramid[j].permute(0, 3, 1, 2)
                x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                self.tensorboard_writer.add_images('bwd_rigid_warp_scale' + str(j), x, n_iter)

        self.fwd_rigid_error_pyramid = [
            image_similarity(args.simi_alpha,
                             self.tgt_view_tile_pyramid[scale],
                             self.fwd_rigid_warp_pyramid[scale])
            for scale in range(args.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            image_similarity(args.simi_alpha, self.src_views_pyramid[scale],
                             self.bwd_rigid_warp_pyramid[scale])
            for scale in range(args.num_scales)
        ]

        if n_iter % 10000 == 0:
            self.fwd_rigid_error_scale = []
            self.bwd_rigid_error_scale = []
            # fwd_rigid_error_pyramid[0]: (8, 3, 128, 416)

            for j in range(len(self.fwd_rigid_error_pyramid)):
                tmp = torch.mean(self.fwd_rigid_error_pyramid[j].permute(0, 3, 1, 2), dim=1, keepdim=True)
                # tmp: (8, 1, 128, 416) in 1st iteration
                self.tensorboard_writer.add_images('fwd_rigid_error_scale' + str(j), tmp, n_iter)
                self.fwd_rigid_error_scale.append(tmp)

            for j in range(len(self.bwd_rigid_error_pyramid)):
                tmp = torch.mean(self.bwd_rigid_error_pyramid[j].permute(0, 3, 1, 2), dim=1, keepdim=True)
                self.tensorboard_writer.add_images('bwd_rigid_error_scale' + str(j), tmp, n_iter)
                self.bwd_rigid_error_scale.append(tmp)

    #####################################################################################################
    """
    def build_flownet(self):

        # output residual flow
        # TODO: non residual mode
        #   make input of the flowNet
        # cat along the color channels
        # shapes: #batch*#src_views, 3+3+3+2+1,h,w

        fwd_flownet_inputs = torch.cat(
            (self.tgt_view_tile_pyramid[0], self.src_views_pyramid[0],
             self.fwd_rigid_warp_pyramid[0], self.fwd_rigid_flow_pyramid[0],
             L2_norm(self.fwd_rigid_error_pyramid[0], dim=1)),
            dim=1)
        bwd_flownet_inputs = torch.cat(
            (self.src_views_pyramid[0], self.tgt_view_tile_pyramid[0],
             self.bwd_rigid_warp_pyramid[0], self.bwd_rigid_flow_pyramid[0],
             L2_norm(self.bwd_rigid_error_pyramid[0], dim=1)),
            dim=1)

        # shapes: # batch
        flownet_inputs = torch.cat((fwd_flownet_inputs, bwd_flownet_inputs),
                                   dim=0)

        # shape: (#batch*2, (3+3+3+2+1)*#src_views, h,w)
        self.resflow = self.flow_net(flownet_inputs)

    def build_full_warp_flow(self):
        # unnormalize the pyramid flow back to pixel metric
        resflow_scaling = []
        # for s in range(self.num_scales):
        #     batch_size, _, h, w = self.resflow[s].shape
        #     # create a scale factor matrix for pointwise multiplication
        #     # NOTE: flow channels x,y
        #     scale_factor = torch.tensor([w, h]).view(1, 2, 1,
        #                                              1).float().to(device)
        #     scale_factor = scale_factor.repeat(batch_size, 1, h, w)
        #     resflow_scaling.append(self.resflow[s] * scale_factor)

        # self.resflow = resflow_scaling

        self.fwd_full_flow_pyramid = [
            self.resflow[s][:self.batch_size * self.num_source,:,:,:] +
            self.fwd_rigid_flow_pyramid[s][:,:,:,:] for s in range(self.num_scales)
        ]
        self.bwd_full_flow_pyramid = [
            self.resflow[s][:self.batch_size * self.num_source,:,:,:] +
            self.bwd_rigid_flow_pyramid[s][:,:,:,:] for s in range(self.num_scales)
        ]

        self.fwd_full_warp_pyramid = [
            flow_warp(self.src_views_pyramid[s], self.fwd_full_flow_pyramid[s])
            for s in range(self.num_scales)
        ]
        self.bwd_full_warp_pyramid = [
            flow_warp(self.tgt_view_tile_pyramid[s],
                      self.bwd_full_flow_pyramid[s])
            for s in range(self.num_scales)
        ]

        self.fwd_full_error_pyramid = [
            image_similarity(self.simi_alpha, self.fwd_full_warp_pyramid[s],
                             self.tgt_view_tile_pyramid[s])
            for s in range(self.num_scales)
        ]
        self.bwd_full_error_pyramid = [
            image_similarity(self.simi_alpha, self.bwd_full_warp_pyramid[s],
                             self.src_views_pyramid[s])
            for s in range(self.num_scales)
        ]
    """

    def build_losses(self):
        """
        # NOTE: geometrical consistency
        if self.train_flow:
            bwd2fwd_flow_pyramid = [
                flow_warp(self.bwd_full_flow_pyramid[s],
                          self.fwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]
            fwd2bwd_flow_pyramid = [
                flow_warp(self.fwd_full_flow_pyramid[s],
                          self.bwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]

            fwd_flow_diff_pyramid = [
                torch.abs(bwd2fwd_flow_pyramid[s] +
                          self.fwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]
            bwd_flow_diff_pyramid = [
                torch.abs(fwd2bwd_flow_pyramid[s] +
                          self.bwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]

            fwd_consist_bound_pyramid = [
                self.geometric_consistency_beta * self.fwd_full_flow_pyramid[s]
                * 2**s for s in range(self.num_scales)
            ]
            bwd_consist_bound_pyramid = [
                self.geometric_consistency_beta * self.bwd_full_flow_pyramid[s]
                * 2**s for s in range(self.num_scales)
            ]
            # stop gradient at maximum opeartions
            fwd_consist_bound_pyramid = [
                torch.max(s,
                          self.geometric_consistency_alpha).clone().detach()
                for s in fwd_consist_bound_pyramid
            ]

            bwd_consist_bound_pyramid = [
                torch.max(s,
                          self.geometric_consistency_alpha).clone().detach()
                for s in bwd_consist_bound_pyramid
            ]

            fwd_mask_pyramid = [(fwd_flow_diff_pyramid[s] * 2**s <
                                 fwd_consist_bound_pyramid[s]).float()
                                for s in range(self.num_scales)]
            bwd_mask_pyramid = [(bwd_flow_diff_pyramid[s] * 2**s <
                                 bwd_consist_bound_pyramid[s]).float()
                                for s in range(self.num_scales)]
        """
        args = self.args

        self.loss_rigid_warp = 0
        self.loss_disp_smooth = 0

        if args.train_flow:
            self.loss_full_warp = 0
            self.loss_full_smooth = 0
            self.loss_geometric_consistency = 0

        for s in range(args.num_scales):
            self.loss_rigid_warp += args.loss_weight_rigid_warp * \
                                    args.num_source / 2 * (
                                            torch.mean(self.fwd_rigid_error_pyramid[s]) +
                                            torch.mean(self.bwd_rigid_error_pyramid[s]))

            #             print(self.loss_disparities[0].size())
            #             print(torch.cat((self.tgt_view_pyramid[3], self.src_views_pyramid[3]), dim=0).size())
            self.loss_disp_smooth += args.loss_weight_disparity_smooth / (2 ** s) * \
                                     smooth_loss(self.loss_depth[s], torch.cat(
                                         (self.tgt_view_pyramid[s], self.src_views_pyramid[s]), dim=0))

            """
            if self.train_flow:
                self.loss_full_warp += self.loss_weight_full_warp * self.num_source / 2 * (
                    torch.sum(
                        torch.mean(self.fwd_full_error_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                        torch.mean(self.bwd_full_error_pyramid[s], 1, True) *
                        bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))

                self.loss_full_smooth += self.loss_weigtht_full_smooth/2**(s+1) *\
                    (flow_smooth_loss(
                        self.fwd_full_flow_pyramid[s], self.tgt_view_tile_pyramid[s]) +
                        flow_smooth_loss(self.bwd_full_flow_pyramid[s], self.src_views_pyramid[s]))

                self.loss_geometric_consistency += self.loss_weight_geometrical_consistency / 2 * (
                    torch.sum(
                        torch.mean(fwd_flow_diff_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                        torch.mean(bwd_flow_diff_pyramid[s], 1, True) *
                        bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))
            """

        self.loss_total = self.loss_rigid_warp + self.loss_disp_smooth

        """
        if self.train_flow:
            print('full warp: {} full_smooth: {}, geo_con:{}'.format(self.loss_full_warp,self.loss_full_smooth,self.loss_geometric_consistency))
            self.loss_total += self.loss_full_warp + \
                self.loss_full_smooth + self.loss_geometric_consistency
        """

    def training_inside_epoch(self):
        global n_iter
        args = self.args

        print("Length of train loader: {}".format(len(self.train_loader)))
        for i, sampled_batch in enumerate(self.train_loader):
            """
            Length of train_loader: num_sequences/4
            Length of sampled_batch: 3
            sampled_batch[i] : [batch_size, channels, height, width]
            """
            start = time.time()

            self.iter_data_preparation(sampled_batch)

            self.build_dispnet()
            self.build_posenet()

            self.build_rigid_warp_flow()

            if args.train_flow:
                self.build_flownet()
                self.build_full_warp_flow()

            self.build_losses()

            """
            if torch.cuda.is_available(): 
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
            """

            self.optimizer.zero_grad()
            self.loss_total.backward()
            self.optimizer.step()

            if n_iter % 100 == 0:
                print('Iteration: {} \t Rigid-warp: {:.4f} \t Disp-smooth: {:.6f}\tTime: {:.3f}'.format(n_iter,
                                                                                                        self.loss_rigid_warp.item(),
                                                                                                        self.loss_disp_smooth.item(),
                                                                                                        time.time() - start))

                self.tensorboard_writer.add_scalar('total_loss', self.loss_total.item(), n_iter)
                self.tensorboard_writer.add_scalar('rigid_warp_loss', self.loss_rigid_warp.item(), n_iter)
                self.tensorboard_writer.add_scalar('disp_smooth_loss', self.loss_disp_smooth.item(), n_iter)

            if n_iter % args.output_ckpt_iter == 0 and n_iter != 0:
                path = '{}/{}_{}'.format(args.ckpt_dir, 'flow' if args.train_flow else 'rigid_depth',
                                         str(n_iter) + '.pth')

                torch.save({
                    'iter': i,
                    'disp_net_state_dict': self.disp_net.state_dict(),
                    'pose_net_state_dict': self.pose_net.state_dict(),
                    'loss': self.loss_total
                }, path)

            n_iter += 1

    def train(self):
        global n_iter
        global device
        args = self.args
        # Sets mode of models to 'train'
        if not args.train_flow:
            self.pose_net.train()
            self.disp_net.train()

        print('Constructing dataset object...')
        self.train_set = SequenceFolder(
            root=args.data_dir,
            seed=args.seed,
            split='train',
            img_height=args.img_height,
            img_width=args.img_width,
            sequence_length=args.sequence_length)

        print('Constructing dataloader object...')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            drop_last=True,
            num_workers=args.data_workers,
            batch_size=args.batch_size,
            pin_memory=True)

        optim_params = [{
            'params': v.parameters(),
            'lr': args.learning_rate
        } for v in self.nets.values()]

        self.optimizer = torch.optim.Adam(
            optim_params,
            betas=(args.momentum, args.beta),
            weight_decay=args.weight_decay)

        print('Starting training for {} epochs...'.format(args.epochs))
        for epoch in range(args.epochs):
            print('-------------------------------EPOCH {}---------------------------------'.format(epoch))

            self.training_inside_epoch()

    @torch.no_grad()
    def test_depth(self):
        pred_depth = None
        args = self.args
        # Sets mode of models to 'eval'
        if not args.train_flow:
            self.pose_net.eval()
            self.disp_net.eval()

        print('Constructing test dataset object...')
        self.test_set = testSequenceFolder(
            root=args.test_dir,
            seed=args.seed,
            split='test',
            img_height=args.img_height,
            img_width=args.img_width,
            sequence_length=args.sequence_length)

        print('Constructing test dataloader object...')
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            shuffle=False,
            drop_last=False,
            num_workers=args.data_workers,
            batch_size=args.batch_size,
            pin_memory=True)

        print("Length of test loader: {}".format(len(self.test_loader)))

        pred_all = []
        for i, sampled_batch in enumerate(self.test_loader):
            """
            Length of test_loader: number of sequences/4
            sampled_batch : [batch_size, channels, height, width]
            """

            start = time.time()

            self.preprocess_test_data(sampled_batch)
            self.build_dispnet()

            pred_depth = self.depth[0]

        return pred_depth
            # pred: (batch_size, height, width)

            # for b in range(sampled_batch.shape[0]):
            #     pred_all.append(pred[b, :, :].cpu().numpy())

        #save_dir_path = args.outputs_dir + os.path.basename(args.ckpt_dir)

        #if not os.path.exists(save_dir_path):
            #os.makedirs(save_dir_path)

        # save_path = save_dir_path + "/rigid__" + str(args.ckpt_index) + '.npy'

        # print("Saving depth predictions to {}".format(save_path))
        # np.save(save_path, pred_all)


if __name__ == '__main__':
    model = NNET()
    x = torch.rand(2, 3, 480, 640)
    out = model(x)
    # print(out.shape)
