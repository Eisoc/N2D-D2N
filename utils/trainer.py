import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.ar_transforms.sp_transfroms import RandomAffineFlow
from utils.depth2flow import all_flows_from_depth_pose
from utils.logger import Logger
from utils.loss_fuctions import Criteria
from utils.utils import build_img_pyramid, L2_norm


class Trainer:
    def __init__(self, model, optimizer, scheduler,train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.global_step = 0
        self.max_step = self.config["epochs"] * len(self.train_loader)
        self.criteria = Criteria(self.config['criteria'])

        self.sp_transform = RandomAffineFlow(
            self.config["st_config"], addnoise=self.config["st_config"]["add_noise"]
        ).to("cuda")

        self.writer = None
        if self.config["verbose"]:
            log_dir = os.path.join(".runs", self.config['usr_msg'])
            self.writer = SummaryWriter(log_dir=log_dir)
            self.logger = Logger(self.writer)

        # Initialize or load checkpoint
        self.saved_epoch, self.saved_step = self._initialize_or_load_checkpoint()

        # Load SENSE encoder
        if self.config["load_SENSE_encoder"]:
            self._load_SENSE_encoder()

    def _save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.epoch}_{self.global_step}_{self.max_step}.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "sheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
            },
            path,
        )

    def _load_checkpoint(self):
        checkpoint = torch.load(self.config["load_path"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {self.config['load_path']}")
        return checkpoint["epoch"] + 1, checkpoint["global_step"]

    def _initialize_or_load_checkpoint(self):
        if self.config["save"]:
            self.save_path = os.path.join(
                self.config["save_path"], "checkpoints", self.config['usr_msg']
            )
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "config.json"), "w") as f:
                json.dump(self.config, f, indent=4)
        if self.config["load_path"] is None:
            return 0, 0
        return self._load_checkpoint()

    def _load_SENSE_encoder(self):
        checkpoint = torch.load(self.config["load_path"])
        self.model.encoder.load_state_dict(checkpoint["model_state_dict"]["encoder"])
        print(f"Loaded SENSE encoder from {self.config['load_path']}")

    def infer(self, tgt_img, ref_imgs, K):
        output = self.model.pose_and_disp(tgt_img, ref_imgs)

        if self.config["depth2flow"]:
            all_flows_from_depth_pose(output, K, min_depth=1, max_depth=5000)
        num_scales = len(output["disp_tgt"])
        tgt_tile_pyramid, ref_pyramid = build_img_pyramid(tgt_img, ref_imgs, num_scales)
        output.update(self.criteria.rigid_flow(tgt_tile_pyramid, ref_pyramid, output, len(ref_imgs)))

        fwd_flownet_inputs = torch.cat(
            (tgt_tile_pyramid[0], ref_pyramid[0],
             output["fwd_rigid_warp_pyramid"][0], output["fwd_rigid_flow_pyramid"][0],
             L2_norm(output["fwd_rigid_error_pyramid"][0], dim=1)),
            dim=1)
        bwd_flownet_inputs = torch.cat(
            (ref_pyramid[0], tgt_tile_pyramid[0],
             output["bwd_rigid_warp_pyramid"][0], output["bwd_rigid_flow_pyramid"][0],
             L2_norm(output["bwd_rigid_error_pyramid"][0], dim=1)),
            dim=1)
        flownet_inputs = torch.cat((fwd_flownet_inputs, bwd_flownet_inputs), dim=0)
        # shape: (#batch*2, (3+3+3+2+1)*#ref_imgs, h,w)
        output.update({"resflow": self.model.flow_net(flownet_inputs)})
        output.update(self.criteria.full_flow(tgt_tile_pyramid, ref_pyramid, output, len(ref_imgs)))
        return output, num_scales

    def _train_step(self, data):
        tgt_img, ref_imgs, K = data
        tgt_img, ref_imgs, K = tgt_img.to("cuda"), [img.to("cuda") for img in ref_imgs], K.to("cuda")

        self.optimizer.zero_grad()

        org_output, num_scales = self.infer(tgt_img, ref_imgs, K)

        augmentation = {"aug_tgt_img": None, "aug_ref_imgs": None, "psd_gt": None, "aug_output": None}
        if self.config["stages"] == 2:
            aug_tgt_img, aug_ref_imgs, psd_gt = self.sp_transform(tgt_img, ref_imgs, org_output)
            aug_output, num_scales = self.infer(aug_tgt_img, aug_ref_imgs, K)
            augmentation = {
                "aug_tgt_img": aug_tgt_img,
                "aug_ref_imgs": aug_ref_imgs,
                "psd_gt": psd_gt,
                "aug_output": aug_output,
            }

        tgt_tile_pyramid, ref_pyramid = build_img_pyramid(tgt_img, ref_imgs, num_scales)
        loss = self.criteria.build_losses(tgt_tile_pyramid, ref_pyramid, org_output, len(ref_imgs), augmentation)

        import pdb; pdb.set_trace()
        loss.backward()

        self.optimizer.step()
        return loss

    def train(self):
        for self.epoch in range(self.config["epochs"]):
            if self.epoch < self.saved_epoch:
                self.global_step += len(self.train_loader)
                continue
            self.model.train()
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                if self.global_step < self.saved_step:
                    self.global_step += 1
                    continue
                loss = self._train_step(data)
                if self.config["verbose"]:
                    self.logger.log(data, loss, {"loss": "train_loss"}, self.global_step)
                self.global_step += 1
                if self.global_step % 1000 == 0:
                    self.scheduler.step()
            self._validation(self.epoch)
            if self.config["save"]:
                self._save_checkpoint()
        if self.config["verbose"]:
            self.writer.close()

    def _validation(self, epoch):
        self.model.eval()
        loss = 0
        for batch_idx, data in enumerate(tqdm(self.val_loader)):
            loss += self._validation_step(data)
        loss /= len(self.val_loader)
        self.logger.log(loss=loss, names={"loss": "validation_loss"}, global_step=epoch)

    def _validation_step(self, data):
        tgt_img, ref_imgs, K = data
        tgt_img, ref_imgs, K = tgt_img.to("cuda"), [img.to("cuda") for img in ref_imgs], K.to("cuda")

        org_output, num_scales = self.infer(tgt_img, ref_imgs, K)
        augmentation = {"aug_tgt_img": None, "aug_ref_imgs": None, "psd_gt": None, "aug_output": None}
        if self.config["stages"] == 2:
            aug_tgt_img, aug_ref_imgs, psd_gt = self.sp_transform(tgt_img, ref_imgs, org_output)
            aug_output, num_scales = self.infer(aug_tgt_img, aug_ref_imgs, K)
            augmentation = {
                "aug_tgt_img": aug_tgt_img,
                "aug_ref_imgs": aug_ref_imgs,
                "psd_gt": psd_gt,
                "aug_output": aug_output,
            }

        tgt_tile_pyramid, ref_pyramid = build_img_pyramid(tgt_img, ref_imgs, num_scales)
        loss = self.criteria.build_losses(tgt_tile_pyramid, ref_pyramid, org_output, len(ref_imgs), augmentation)
        return loss
