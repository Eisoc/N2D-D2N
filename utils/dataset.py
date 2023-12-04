import os
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip

from .utils import list_subdirs, get_P_rect, load_image

KITTI_RAW_DIR = "/home/tuslam/sa/data/kitti"
KITTI_MULTIVIEW_DIR = "/root/autodl-tmp/data/multiview"


def make_data_loader(config, stages):
    if config["dataset"] == "raw":
        train_data, val_data = make_sequence_dataset(config["split"])  # [[tar_im_path, [ref_im_paths]], calib_path]
    elif config["dataset"] == "multiview":
        train_data, val_data = make_multiview_dataset()

    # WARN: train on small set
    train_data = train_data[:10000]

    print("{} samples found for joint training.".format(len(train_data)))
    print("{} samples found for joint valing.".format(len(val_data)))

    if stages == 1:
        train_set = MapDataset(train_data, (320, 1024), [RandomHorizontalFlip(), RandomCrop((200, 640))])
    elif stages == 2:
        train_set = MapDataset(train_data, (320, 1024), [])
    else:
        raise Exception("stages must be 1 or 2")
    val_set = MapDataset(val_data, (320, 1024), [])

    train_loader = DataLoader(
        train_set,
        batch_size=config["batchsize"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batchsize"],
        shuffle=False,
        drop_last=True,
    )

    return train_loader, val_loader


def make_sequence_dataset(split=-1):
    def make_onedrive_dataset(root, calib_path):
        img_dir = os.path.join(root, "image_02/data")
        img_path = []
        for path in os.scandir(img_dir):
            if path.name.endswith(".png"):
                img_path.append(path.path)
        img_path.sort()

        n = len(img_path)  # The number of random integers you want to generate

        dataset = [[[img_path[i], [img_path[i - 1], img_path[i + 1]]], calib_path] for i in range(n - 1)]

        return dataset

    def make_oneday_dataset(root):
        dataset = []
        calib_path = os.path.join(root, "calib_cam_to_cam.txt")
        assert os.path.exists(calib_path), calib_path
        subdirs = list_subdirs(root)

        for subdir in subdirs:
            dataset += make_onedrive_dataset(subdir, calib_path)

        return dataset

    subdirs = list_subdirs(KITTI_RAW_DIR)
    all_data = []
    for subdir in subdirs:
        all_data += make_oneday_dataset(subdir)

    if split > 0:
        n = len(all_data)
        if split < 1:
            val_split = int(split * n)  # Index to split the list
        else:
            val_split = split
        # Randomly select elements for the smaller portion
        val_data = random.sample(all_data, int(val_split))

        # Remaining elements for the larger portion
        train_data = [element for element in all_data if element not in val_data]

    else:
        val_data = []
        train_data = all_data

    return train_data, val_data


def make_multiview_dataset():
    def one_group_dataset(root, group_idx):
        calib_path = os.path.join(root, "calib_cam_to_cam", f"{group_idx:06}.txt")
        assert os.path.exists(calib_path), calib_path
        n = 21
        img_path = []
        for i in range(n):
            if os.path.exists(os.path.join(root, "image_2", f"{group_idx:06}_{i:02}.png")):
                img_path.append(os.path.join(root, "image_2", f"{group_idx:06}_{i:02}.png"))
            else:
                print(group_idx, i)
                break
        # img_path = [os.path.join(root, "image_2", f"{group_idx:06}_{i:02}.png") for i in range(n)]
        n = len(img_path)
        dataset = [[[img_path[i], [img_path[i - 1], img_path[i + 1]]], calib_path] for i in range(n - 1)]

        return dataset

    train_data_path = os.path.join(KITTI_MULTIVIEW_DIR, "training")
    val_data_path = os.path.join(KITTI_MULTIVIEW_DIR, "testing")

    train_data = []
    val_data = []
    for group_idx in range(200):
        train_data += one_group_dataset(train_data_path, group_idx)
        val_data += one_group_dataset(val_data_path, group_idx)

    return train_data, val_data


class MapDataset(Dataset):
    def __init__(self, datalist, size, transforms):
        self.data = datalist
        self.h, self.w = size
        # use original image in the first stage
        self.transforms = Compose(transforms + [Resize(size)])

    def __getitem__(self, idx):
        [tar_path, ref_paths], calib_path = self.data[idx]

        tar_img = load_image(tar_path)
        h, w = tar_img.shape[-2:]

        tar_img = self.transforms(tar_img)
        ref_imgs = [self.transforms(load_image(ref_path)) for ref_path in ref_paths]

        # load intrisinc matrix
        K = get_P_rect(calib_path, self.h / h, self.w / w)[:, :3]

        return tar_img, ref_imgs, K.astype(np.float32)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pass
