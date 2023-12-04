import os

import numpy as np
import torch
from torch import float32
from torchvision.io import read_image


def load_image(img_path):
    img = read_image(img_path).to(float32)
    return img / 255.0


def list_subdirs(rootdir):
    subdirs = []
    for path in os.scandir(rootdir):
        if path.is_dir():
            subdirs.append(path.path)
    return subdirs


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def get_P_rect(calib_file, zoom_x, zoom_y):
    filedata = read_calib_file(calib_file)
    P_rect = np.reshape(filedata["P_rect_02"], (3, 4))
    P_rect[0] *= zoom_x
    P_rect[1] *= zoom_y
    return P_rect


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def scale_pyramid(img, num_scales):
    # shape of img: batch_size,channels,h,w
    if img is None:
        return None
    else:
        scaled_imgs = [img]
        # TODO: Assume the shape of image is [#channels, #rows, #cols ]
        h, w = img.shape[-2:]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = int(h / ratio)
            nw = int(w / ratio)
            scaled_img = torch.nn.functional.interpolate(img, size=(nh, nw))
            scaled_imgs.append(scaled_img)
    # shape: #scales, # batch, #chnl, h, w
    return scaled_imgs


def build_img_pyramid(tgt_img, ref_imgs, num_scales):
    # 　shape:  #scale, #batch, #chnls, h,w
    tgt_img_pyramid = scale_pyramid(tgt_img, num_scales)
    # 　shape:  #scale, #batch*#ref_imgs, #chnls,h,w
    n = len(ref_imgs)
    tgt_tile_pyramid = [
        tgt_img_pyramid[scale].repeat(n, 1, 1, 1)
        for scale in range(num_scales)
    ]

    # 　shape:  # scale,#batch*#ref_imgs, # chnls, h,w
    ref_concat = torch.cat(ref_imgs, 0)
    ref_pyramid = scale_pyramid(ref_concat, num_scales)
    return tgt_tile_pyramid, ref_pyramid
def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
    return l2_norm
def gradient_x(img):
    return img[:, :, :, :-1]-img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :]-img[:, :, 1:, :]
