import torch
from flow_vis import flow_to_color


class Logger:
    def __init__(self, writer) -> None:
        self.writer = writer

    def _log_images(self, depth_eval, flow_eval, data, global_step):
        tgt_img, ref_imgs, K = data
        if depth_eval is not None:
            self.writer.add_images(
                "depth_warp_imgs",
                torch.cat(
                    (
                        tgt_img,
                        ref_imgs[0],
                        depth_eval["warped_imgs"][0][0],
                        ref_imgs[1],
                        depth_eval["warped_imgs"][0][1],
                    ),
                    2,
                ),
                global_step,
            )
            self.writer.add_images(
                "depth_occ_imgs",
                torch.cat(
                    (
                        self._normalize_tensor(torch.log(depth_eval["depths"][0])).unsqueeze(1),
                        depth_eval["occlusions"][0].unsqueeze(1),
                    ),
                    2,
                ),
                global_step,
            )

        if flow_eval is not None:
            self.writer.add_images(
                "flow_warp_imgs",
                torch.cat(
                    (
                        tgt_img,
                        ref_imgs[0],
                        flow_eval["warped_imgs"][0][0],
                        ref_imgs[1],
                        flow_eval["warped_imgs"][0][1],
                    ),
                    2,
                ),
                global_step,
            )
            self.writer.add_images(
                "flow_imgs",
                torch.cat(
                    (
                        self._normalize_tensor(torch.norm(flow_eval["flows"][0][0], dim=1)).unsqueeze(1),
                        self._normalize_tensor(torch.norm(flow_eval["flows"][0][1], dim=1)).unsqueeze(1),
                    ),
                    2,
                ),
                global_step,
            )
            self.writer.add_images(
                "occ_of_flow_imgs",
                torch.cat(
                    (
                        flow_eval["occlusions"][0][0].unsqueeze(1),
                        flow_eval["occlusions"][0][1].unsqueeze(1),
                    ),
                    2,
                ),
                global_step,
            )

    def _log_scalars(self, depth_eval, flow_eval, losses, global_step):
        if depth_eval is not None:
            depth_scales = len(depth_eval["warp_losses"])
            self.writer.add_scalars(
                "depth_warp_losses",
                {f"scale {i}": v.item() for i, v in zip(range(depth_scales), depth_eval["warp_losses"])},
                global_step,
            )
            self.writer.add_scalars(
                "depth_occ_regs",
                {f"scale {i}": v.item() for i, v in zip(range(depth_scales), depth_eval["occ_regs"])},
                global_step,
            )
            self.writer.add_scalar("depth_loss", losses["depth"].item(), global_step)

        if flow_eval is not None:
            flow_scales = len(flow_eval["warp_losses"])
            self.writer.add_scalars(
                "flow_warp_losses",
                {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_eval["warp_losses"])},
                global_step,
            )
            self.writer.add_scalars(
                "flow_SSIM_losses",
                {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_eval["SSIM_losses"])},
                global_step,
            )
            self.writer.add_scalars(
                "flow_smooth_losses",
                {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_eval["smooth_losses"])},
                global_step,
            )
            self.writer.add_scalars(
                "flow_mask_regs",
                {f"scale {i}": v.item() for i, v in zip(range(flow_scales), flow_eval["occ_regs"])},
                global_step,
            )

            self.writer.add_scalar("flow_aug_loss", flow_eval["aug_loss"].item(), global_step)

            self.writer.add_scalar("flow_loss", losses["flow"].item(), global_step)

        self.writer.add_scalar("loss", losses["total"].item(), global_step)

    def log(self, data=None, loss=None, names=None, global_step=None):
        self.writer.add_scalar(names["loss"], loss, global_step)

    def _normalize_tensor(self, tensor):
        min_value = tensor.min()
        max_value = tensor.max()
        normalized_tensor = (tensor - min_value) / (max_value - min_value)
        return normalized_tensor
