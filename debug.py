"""

"""
from logging import raiseExceptions
import sys
import os
import time
import datetime
import random

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm

from models.model import Net
from utils.dataset import MapDataset, make_sequence_dataset
from utils.arguments import parse_args
from utils.loss_fuctions import depth_criteria, flow_criteria, flow_reconstruction_loss, flow_smoothness_loss
from utils.visualization import tensor2array

torch.autograd.set_detect_anomaly(True)


def make_data_loader(args):
    train_data, val_data = make_sequence_dataset(args.split, args.n_ref)  # [[tar_im_path, [ref_im_paths]], calib_path]

    print("{} samples found for joint training.".format(len(train_data)))
    print("{} samples found for joint valing.".format(len(val_data)))

    train_set = MapDataset(train_data, (320, 1024))
    val_set = MapDataset(val_data, (320, 1024))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, val_loader


def test(model, test_loader, args):
    def load_checkpoint():
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"load checkpoint from {args.load_path}")

    # verbose
    if args.verbose:
        writer = SummaryWriter()

    global_step = 0
    max_step = args.epochs * len(test_loader)

    load_checkpoint()

    model.eval()
        for batch_idx, data in enumerate(tqdm(test_loader)):
            tar_img, ref_imgs, K = data
            tar_img = tar_img.to("cuda")
            ref_imgs = [img.to("cuda") for img in ref_imgs]
            K = K.to("cuda")

            disp, pose, flow_occ = model(tar_img, ref_imgs)

            # build synthesised image from depth prediction, flow prediction and pose prediction
            depth_losses, depth_mask_regs, depth_warped_results, depth_diff_results, depth_results = depth_criteria(
                tar_img, ref_imgs, K, disp, pose, args
            )
            depth_scales = len(depth_losses)
            depth_loss = 0
            depth_mask_reg = 0
            for i in range(depth_scales):
                depth_loss += depth_losses[i] / depth_scales
                depth_mask_reg += depth_mask_regs[i] / depth_scales

            (
                flow_reconstruction_losses,
                flow_smoothness_losses,
                flow_mask_regs,
                flow_warped_results,
                flow_diff_results,
            ) = flow_criteria(tar_img, ref_imgs, flow_occ)
            flow_scales = len(flow_reconstruction_losses)
            flow_loss = 0
            flow_mask_reg = 0
            flow_smoothness_loss = 0
            for i in range(flow_scales):
                flow_loss += flow_reconstruction_losses[i]
                flow_smoothness_loss += flow_smoothness_losses[i]
                flow_mask_reg += flow_mask_regs[i]

            # loss = (
            #     depth_loss + flow_loss + args.depth_mask_weight * depth_mask_reg + args.flow_mask_weight * flow_mask_reg
            # )
            loss = flow_loss + args.flow_mask_weight * flow_mask_reg + 0.5 * flow_smoothness_loss

            # verbose
            if args.verbose:
                if batch_idx % 100 == 0:
                    depth_im_np = np.stack([tensor2array(depth_results[0][i], 3000) for i in range(tar_img.shape[0])])
                    depth_im = torch.from_numpy(depth_im_np)
                    writer.add_images(
                        "depth_imgs",
                        torch.cat(
                            (
                                tar_img.detach().cpu(),
                                ref_imgs[0].detach().cpu(),
                                depth_im,
                                depth_warped_results[0][0],
                            ),
                            2,
                        ),
                        global_step,
                    )
                    writer.add_images(
                        "flow_imgs",
                        torch.cat((tar_img.detach().cpu(), ref_imgs[0].detach().cpu(), flow_warped_results[0][0]), 2),
                        global_step,
                    )
                writer.add_scalars(
                    "depth_losses",
                    {f"scale {i}": v.item() for i, v in zip(range(depth_scales), depth_losses)},
                    global_step,
                )
                writer.add_scalars(
                    "depth_mask_regs",
                    {f"scale {i}": v.item() for i, v in zip(range(depth_scales), depth_mask_regs)},
                    global_step,
                )
                writer.add_scalars(
                    "flow_reconstruction_losses",
                    {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_reconstruction_losses)},
                    global_step,
                )
                writer.add_scalars(
                    "flow_smoothness_losses",
                    {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_smoothness_losses)},
                    global_step,
                )
                writer.add_scalars(
                    "flow_mask_regs",
                    {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_mask_regs)},
                    global_step,
                )

                writer.add_scalar("depth_avg_loss", depth_loss.item(), global_step)
                writer.add_scalar("flow_loss", flow_loss.item(), global_step)
                writer.add_scalar("loss", loss.item(), global_step)

            global_step += 1

    # versobe
    if args.versobe:
        writer.close()


def main(args):
    test_loader, _ = make_data_loader(args)

    model = Net(args)
    model.to("cuda")
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    test(model, test_loader, args)


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Use following parameters:")
    for k, v in vars(args).items():
        print("{}\t{}".format(k, v))
    print("=======================================\n")

    main(args)
