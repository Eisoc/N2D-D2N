"""
loss fucntions for MyNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .depth2flow import disp_to_depth
from .inverse_warp import inverse_warp
from .utils import gradient_y, gradient_x


class DepthCriteria:
    """[TODO:description]

    Attributes:
        min_depth: [TODO:attribute]
        max_depth: [TODO:attribute]
        rotation_mode: [TODO:attribute]
        padding_mode: [TODO:attribute]
        reconstruction_losses: [TODO:attribute]
        mask_regs: [TODO:attribute]
        warped_imgs: [TODO:attribute]
        depths: depth of target image on different scales
    """

    def __init__(self, config, rotation_mode="euler", padding_mode="zeros") -> None:
        self.min_depth = config["min_depth"]
        self.max_depth = config["max_depth"]
        self.rotation_mode = rotation_mode
        self.padding_mode = padding_mode

        # self._reset_properties()

    # def _reset_properties(self):
    #     self.reconstruction_losses = []
    #     self.smooth_losses = []
    #     self.mask_regs = []
    #     self.warped_imgs = []
    #     self.depths = []

    def evaluate(self, tar_img, ref_imgs, intrinsics, org_output, aug_output=None):
        # self._reset_properties()
        org_disp = org_output["disp"]
        org_pose = org_output["pose"]
        l = len(org_disp)
        # calcualte losses for each scale
        reconstruction_losses, mask_regs, smooth_losses, depths, occs, warped_imgs = [], [], [], [], [], []
        for i in range(l):
            disp_scaled, occ = org_disp[i][:, 0], org_disp[i][:, 1]  # (bs,w,h)
            _, d = disp_to_depth(disp_scaled, self.min_depth, self.max_depth)

            loss, warped, diff = self._reconstruction_loss(tar_img, ref_imgs, intrinsics, d, occ, org_pose)
            mask_reg = self._mask_regularization(occ)
            smooth_loss = self._smooth_loss(d)

            reconstruction_losses.append(loss)
            mask_regs.append(mask_reg)
            smooth_losses.append(smooth_loss)

            depths.append(d.detach().cpu())
            occs.append(occ.detach().cpu())
            warped_imgs.append(warped)

        return {
            "warp_losses": reconstruction_losses,
            "smooth_losses": smooth_losses,
            "occ_regs": mask_regs,
            "warped_imgs": warped_imgs,
            "depths": depths,
            "occlusions": occs,
        }

    def _mask_regularization(self, occ):
        zeros_var = torch.zeros_like(occ)
        mask_reg = nn.functional.binary_cross_entropy(occ, zeros_var)

        return mask_reg

    def _reconstruction_loss(self, tar_img, ref_imgs, intrinsics, depth, occ, pose):
        assert pose.size(1) == len(ref_imgs)

        reconstruction_loss = 0
        b, h, w = depth.size()
        downscale = tar_img.size(2) / h

        tar_img_scaled = F.interpolate(tar_img, (h, w), mode="area")
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode="area") for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(
                ref_img, depth, current_pose, intrinsics_scaled, self.rotation_mode, self.padding_mode
            )
            if valid_points.sum() / (valid_points.shape[1] * valid_points.shape[2]) < 0.5:
                print("Too few valid points.")
            diff = tar_img_scaled - (
                ref_img_warped
                * valid_points.unsqueeze(1).float()
                * (1 - occ.unsqueeze(1))
                # + tar_img_scaled * occ.unsqueeze(1)
            )

            reconstruction_loss += diff.abs().mean()

            assert (reconstruction_loss == reconstruction_loss).item() == 1

            warped_imgs.append(ref_img_warped.detach().cpu())
            diff_maps.append(diff.detach().cpu())

        return reconstruction_loss, warped_imgs, diff_maps

    def _smooth_loss(self, depth):
        def gradient(pred):
            D_dy = pred[:, 1:] - pred[:, :-1]
            D_dx = pred[:, :, 1:] - pred[:, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(depth)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss = dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()

        return loss


class FlowCriteria:
    """[TODO:description]

    Attributes:
        reconstruction_losses: [TODO:attribute]
        smoothness_losses: [TODO:attribute]
        mask_regs: [TODO:attribute]
        warped_imgs: [TODO:attribute]
        flows: flows of reference images with respect to target image on different scales
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, tar_img, ref_imgs, org_output, augmentation=None):
        n = len(ref_imgs)
        scales = len(org_output["flow"][0])

        reconstruction_losses = []
        SSIM_losses = []
        smooth_losses = []
        census_losses = []
        occ_regs = []

        all_flows = []
        all_occs = []
        warped_imgs = []

        for s in range(scales):
            tgt_img_scaled = None

            rec_loss_one_scale = 0
            SSIM_loss_one_scale = 0
            smooth_loss_one_scale = 0
            census_loss_one_scale = 0
            occ_reg_one_scale = 0

            flows_one_scale = []
            occs_oce_scale = []
            warped_imgs_one_scale = []

            for i in range(n):
                flow = org_output["flow"][i][s]
                # occ = F.softmax(org_output["occ"][i][s], dim=1)[:, 1, :, :]
                occ = org_output["occ"][i][s].squeeze(1)

                b, h, w = occ.size()
                if tgt_img_scaled is None:
                    tgt_img_scaled = F.interpolate(tar_img, (h, w), mode="area")
                ref_img_scaled = F.interpolate(ref_imgs[i], (h, w), mode="area")

                ref_img_warped, valid_points = self._warp_image(ref_img_scaled, flow)

                reconstruction_loss, diff = self._reconstruction_loss(tgt_img_scaled, ref_img_warped, valid_points, occ)
                SSIM_loss = self._SSIM_loss(tgt_img_scaled, ref_img_warped, occ)
                smooth_loss = self._smooth_loss(flow)
                census_loss = self._census_loss(tgt_img_scaled, ref_img_warped, occ)
                occ_reg = self._occ_regularization(occ)

                rec_loss_one_scale += reconstruction_loss
                SSIM_loss_one_scale += SSIM_loss
                smooth_loss_one_scale += smooth_loss
                census_loss_one_scale += census_loss
                occ_reg_one_scale += occ_reg

                flows_one_scale.append(flow.detach().cpu())
                occs_oce_scale.append(occ.detach().cpu())
                warped_imgs_one_scale.append(ref_img_warped.detach().cpu())

            reconstruction_losses.append(rec_loss_one_scale)
            SSIM_losses.append(SSIM_loss_one_scale)
            smooth_losses.append(smooth_loss_one_scale)
            census_losses.append(census_loss_one_scale)
            occ_regs.append(occ_reg_one_scale)

            all_flows.append(flows_one_scale)
            all_occs.append(occs_oce_scale)
            warped_imgs.append(warped_imgs_one_scale)

        aug_loss = None
        if augmentation is not None:
            aug_loss = self._augmentation_loss(augmentation)

        return {
            "warp_losses": reconstruction_losses,
            "smooth_losses": smooth_losses,
            "census_losses": census_losses,
            "SSIM_losses": SSIM_losses,
            "occ_regs": occ_regs,
            "warped_imgs": warped_imgs,
            "flows": all_flows,
            "occlusions": all_occs,
            "aug_loss": aug_loss,
        }

    def _warp_image(self, image, flow):
        b, _, h, w = flow.size()

        flow = flow.permute((0, 2, 3, 1))

        meshy, meshx = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack([torch.stack((meshx, meshy), 2).cuda()] * b)  # [x,y] coordinate
        cor = grid + flow
        cor = cor / torch.tensor([(w - 1) / 2, (h - 1) / 2]).cuda() - 1  # normalization

        warped_ref_img = F.grid_sample(image, cor, padding_mode="zeros")
        valid_points = cor.abs().max(dim=-1)[0] <= 1

        return warped_ref_img, valid_points

    def _occ_regularization(self, occ):
        zeros_var = torch.zeros_like(occ)
        occ_reg = nn.functional.binary_cross_entropy(occ, zeros_var)

        return occ_reg

    def _reconstruction_loss(self, tgt_img, warped_img, valid_points, occ):
        diff = tgt_img - (
            warped_img
            * valid_points.unsqueeze(1).float()
            * (1 - occ.unsqueeze(1))
            # + tgt_img_scaled * occ
        )

        reconstruction_loss = diff.abs().mean()
        assert (reconstruction_loss == reconstruction_loss).item() == 1

        return reconstruction_loss, diff

    def _smooth_loss(self, flow):
        """Calculate the smoothness loss for a given optical flow."""

        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        # Calculate the difference between each pixel and its neighboring pixels
        dx, dy = gradient(flow)
        # dx2, dxdy = gradient(dx)
        # dydx, dy2 = gradient(dy)
        # loss = dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        loss=dx.abs().mean()+dy.abs().mean()
        return loss

    def _augmentation_loss(self, augmentation):
        n = len(augmentation["aug_ref_imgs"])

        loss = 0
        for i in range(n):
            flow_t_pred = augmentation["aug_output"]["flow"][i][0]
            flow_t = augmentation["psd_gt"]["flow"][i]
            # noc_t = 1 - F.softmax(augmentation["psd_gt"]["occ"][i], dim=1)[:, 1:, :, :]
            noc_t = 1 - augmentation["psd_gt"]["occ"][i].squeeze(1)

            l_atst = ((flow_t_pred - flow_t).abs() + 0) ** 1
            l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += l_atst

        return loss

    def _SSIM_loss(self, tgt_img, warped_img, occ, md=1):
        patch_size = 2 * md + 1
        C1 = 0.01**2
        C2 = 0.03**2

        mu_x = nn.AvgPool2d(patch_size, 1, 0)(tgt_img * occ.unsqueeze(1))
        mu_y = nn.AvgPool2d(patch_size, 1, 0)(warped_img * occ.unsqueeze(1))
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(patch_size, 1, 0)(tgt_img * tgt_img) - mu_x_sq
        sigma_y = nn.AvgPool2d(patch_size, 1, 0)(warped_img * warped_img) - mu_y_sq
        sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(tgt_img * warped_img) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        dist = torch.clamp((1 - SSIM) / 2, 0, 1).mean()
        return dist

    def _census_loss(self, tgt_img, warped_img, occ, patch_size=7):
        """Compares the similarity of the census transform of two images."""
        tgt_census_img = census_transform(tgt_img, patch_size)
        warped_census_img = census_transform(warped_img, patch_size)

        hamming = soft_hamming(tgt_census_img, warped_census_img)

        # Set borders of mask to zero to ignore edge effects.
        padded_mask = zero_mask_border(1 - occ, patch_size).detach()
        diff = abs_robust_loss(hamming)
        diff *= padded_mask.unsqueeze(1)
        diff_sum = diff.sum()
        loss_mean = diff_sum / (padded_mask.sum() + 1e-6)
        return loss_mean


def zero_mask_border(mask, patch_size):
    """Used to ignore border effects from census_transform."""
    mask_padding = patch_size // 2
    mask = mask[:, mask_padding:-mask_padding, mask_padding:-mask_padding]
    return F.pad(
        mask,
        (mask_padding, mask_padding, mask_padding, mask_padding, 0, 0),
    )


def census_transform(image, patch_size):
    """The census transform as described by DDFlow."""
    b, c, h, w = image.shape

    intensities = T.Grayscale()(image) * 255

    # Using nn.functional.conv2d for patch extraction
    out_channels = patch_size * patch_size
    w = torch.eye(out_channels).reshape((1, patch_size, patch_size, out_channels))
    weights = w.float().permute(3, 0, 1, 2).to(image.device)
    neighbors = F.conv2d(intensities, weights, stride=1, padding=patch_size // 2)

    diff = neighbors - intensities
    diff_norm = diff / torch.sqrt(0.81 + diff**2)

    return diff_norm


def soft_hamming(tgt_img, warped_img, thresh=0.1):
    """A soft hamming distance between tensor a_bhwk and tensor b_bhwk."""
    sq_dist = (tgt_img - warped_img) ** 2
    soft_thresh_dist = sq_dist / (thresh + sq_dist)
    return soft_thresh_dist.sum(dim=1, keepdim=True)


def abs_robust_loss(diff, eps=0.01, q=0.4):
    """The so-called robust loss used by DDFlow."""
    return torch.pow(torch.abs(diff) + eps, q)

#     def _ternary_loss(self, im1, im2_warped, occ, max_distance=1):
#         patch_size = 2 * max_distance + 1
#
#         def _ternary_transform(image):
#             intensities = T.rgb_to_grayscale(image) * 255
#             # patches = tf.extract_image_patches( # fix rows_in is None
#             #    intensities,
#             #    ksizes=[1, patch_size, patch_size, 1],
#             #    strides=[1, 1, 1, 1],
#             #    rates=[1, 1, 1, 1],
#             #    padding='SAME')
#             out_channels = patch_size * patch_size
#             w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
#             weights = tf.constant(w, dtype=tf.float32)
#             weights = torch.eye()
#             patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding="SAME")
#
#             transf = patches - intensities
#             transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
#             return transf_norm
#
#         def _hamming_distance(t1, t2):
#             dist = tf.square(t1 - t2)
#             dist_norm = dist / (0.1 + dist)
#             dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
#             return dist_sum
#
#         t1 = _ternary_transform(im1)
#         t2 = _ternary_transform(im2_warped)
#         dist = _hamming_distance(t1, t2)
#
#         transform_mask = create_mask(occ, [[max_distance, max_distance], [max_distance, max_distance]])
#         return charbonnier_loss(dist, occ * transform_mask)
#
#
# def flow_smoothness_loss(flow_occ):
#     """Calculate the smoothness loss for a given optical flow."""
#
#     # Calculate the difference between each pixel and its neighboring pixels
#     def one_scale(idx):
#         refs = len(flow_occ)
#         dx, dy = 0, 0
#         for i in range(refs):
#             flow = flow_occ[i][0][idx]
#             dx += torch.mean(torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :]))
#             dy += torch.mean(torch.abs(flow[:, :-1, :, :] - flow[:, 1:, :, :]))
#         return dx + dy
#
#     losses = []
#     scales = len(flow_occ[0][0])
#     for i in range(scales):
#         loss = one_scale(i)
#         losses.append(loss)
#
#     return losses
#
#
# def calculate_warp_loss(target, reference, opt):
#     """Calculate the warp loss for a given target image, reference image, and optical flow."""
#     # Generate a grid of coordinates in the target image
#     grid_y, grid_x = torch.meshgrid(torch.arange(target.size(2)), torch.arange(target.size(3)))
#     grid = torch.stack((grid_x, grid_y), 2).float()
#
#     # Use the optical flow to calculate the corresponding coordinates in the reference image
#     flow = grid + opt
#     warped = F.grid_sample(reference, flow)
#
#     # Calculate the mean squared error between the warped image and the target image
#     return F.mse_loss(warped, target)
#
#
# def flow_consistency_loss(opt, opt_back):
#     """Calculate the consistency loss for a given forward and backward optical flow."""
#     # Generate a grid of coordinates in the target image
#     grid_y, grid_x = torch.meshgrid(torch.arange(opt.size(2)), torch.arange(opt.size(3)))
#     grid = torch.stack((grid_x, grid_y), 2).float()
#
#     # Use the forward flow to calculate the corresponding coordinates in the reference image,
#     # and then use the backward flow to map these coordinates back to the target image
#     flow = grid + opt
#     flow_back = grid + F.grid_sample(opt_back, flow)
#
#     # The forward flow and the backward flow should be inverse to each other,
#     # so the difference between the original coordinates and the coordinates mapped back should be minimal
#     return F.mse_loss(flow_back, grid)

class Criteria:
    def __init__(self, args) -> None:
        self.args = args
        self._set_hyperparameters()

    def _set_hyperparameters(self):
        self.alpha = torch.tensor(self.args['alpha']).float().to("cuda")
        self.geometric_consistency_beta = torch.tensor(self.args["geometric_consistency_beta"]).float().to("cuda")
        self.geometric_consistency_alpha = torch.tensor(self.args["geometric_consistency_alpha"]).float().to("cuda")
        self.loss_weight_rigid_warp = torch.tensor(self.args["loss_weight_rigid_warp"]).float().to("cuda")
        self.loss_weight_disparity_smooth = torch.tensor(self.args["loss_weight_disparity_smooth"]).float().to("cuda")
        self.loss_weight_full_smooth = torch.tensor(self.args["loss_weight_full_smooth"]).float().to("cuda")
        self.loss_weight_full_warp = torch.tensor(self.args["loss_weight_full_warp"]).float().to("cuda")
        self.loss_weight_geometrical_consistency = torch.tensor(
            self.args["loss_weight_geometrical_consistency"]).float().to("cuda")
        self.loss_weight_census = torch.tensor(self.args["loss_weight_census"]).float().to("cuda")

    def rigid_flow(self, tgt_tile_pyramid, ref_pyramid, org_output, n):
        """

        @param tgt_tile_pyramid: [(bs*#refs, 3, h, w)...]
        @param ref_pyramid:
        @param org_output:
        @param aug_output:
        @return:
        """
        num_scales = len(org_output["flow_fwd"][0])
        fwd_flow = org_output["flow_fwd"]
        bwd_flow = org_output["flow_bwd"]

        # [(bs*n, 2, h, w), ...]
        fwd_rigid_flow_pyramid = [torch.cat([fwd_flow[i][s] for i in range(n)]) for s in range(num_scales)]
        bwd_rigid_flow_pyramid = [torch.cat([bwd_flow[i][s] for i in range(n)]) for s in range(num_scales)]

        fwd_rigid_warp_pyramid = [self._warp_image(ref_pyramid[scale], fwd_rigid_flow_pyramid[scale]) for scale in
                                  range(num_scales)]

        bwd_rigid_warp_pyramid = [self._warp_image(tgt_tile_pyramid[scale], bwd_rigid_flow_pyramid[scale])
                                  for scale in range(num_scales)]

        fwd_SSIM_pyramid = [self._SSIM_loss(tgt_tile_pyramid[scale], fwd_rigid_warp_pyramid[scale])
                            for scale in range(num_scales)]
        bwd_SSIM_pyramid = [self._SSIM_loss(ref_pyramid[scale], bwd_rigid_warp_pyramid[scale])
                            for scale in range(num_scales)]

        fwd_recon_pyramid = [self._reconstruction_loss(tgt_tile_pyramid[scale], fwd_rigid_warp_pyramid[scale])
                             for scale in range(num_scales)]
        bwd_recon_pyramid = [self._reconstruction_loss(ref_pyramid[scale], bwd_rigid_warp_pyramid[scale])
                             for scale in range(num_scales)]

        fwd_rigid_error_pyramid = [
            self.alpha * fwd_SSIM_pyramid[scale] + (1 - self.alpha) * fwd_recon_pyramid[scale]
            for scale in range(num_scales)
        ]
        bwd_rigid_error_pyramid = [
            self.alpha * bwd_SSIM_pyramid[scale] + (1 - self.alpha) * bwd_recon_pyramid[scale]
            for scale in range(num_scales)
        ]

        return {
            "fwd_rigid_flow_pyramid": fwd_rigid_flow_pyramid,
            "bwd_rigid_flow_pyramid": bwd_rigid_flow_pyramid,
            "fwd_rigid_warp_pyramid": fwd_rigid_warp_pyramid,
            "bwd_rigid_warp_pyramid": bwd_rigid_warp_pyramid,
            "fwd_rigid_error_pyramid": fwd_rigid_error_pyramid,
            "bwd_rigid_error_pyramid": bwd_rigid_error_pyramid,
        }

    def full_flow(self, tgt_tile_pyramid, ref_pyramid, org_output, n):
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
        bs = tgt_tile_pyramid[0].shape[0] // n
        num_scales = len(org_output["flow_fwd"][0])

        fwd_full_flow_pyramid = [
            org_output["resflow"][s][:bs * n, :, :, :] +
            org_output["fwd_rigid_flow_pyramid"][s][:, :, :, :] for s in range(num_scales)
        ]
        bwd_full_flow_pyramid = [
            org_output["resflow"][s][:bs * n, :, :, :] +
            org_output["bwd_rigid_flow_pyramid"][s][:, :, :, :] for s in range(num_scales)
        ]

        fwd_full_warp_pyramid = [
            self._warp_image(ref_pyramid[s], fwd_full_flow_pyramid[s])
            for s in range(num_scales)
        ]
        bwd_full_warp_pyramid = [
            self._warp_image(tgt_tile_pyramid[s], bwd_full_flow_pyramid[s])
            for s in range(num_scales)
        ]

        fwd_SSIM_pyramid = [self._SSIM_loss(tgt_tile_pyramid[scale], fwd_full_warp_pyramid[scale])
                            for scale in range(num_scales)]
        bwd_SSIM_pyramid = [self._SSIM_loss(ref_pyramid[scale], bwd_full_warp_pyramid[scale])
                            for scale in range(num_scales)]

        fwd_recon_pyramid = [self._reconstruction_loss(tgt_tile_pyramid[scale], fwd_full_warp_pyramid[scale])
                             for scale in range(num_scales)]
        bwd_recon_pyramid = [self._reconstruction_loss(ref_pyramid[scale], bwd_full_warp_pyramid[scale])
                             for scale in range(num_scales)]

        fwd_full_error_pyramid = [
            self.alpha * fwd_SSIM_pyramid[scale] + (1 - self.alpha) * fwd_recon_pyramid[scale]
            for scale in range(num_scales)
        ]
        bwd_full_error_pyramid = [
            self.alpha * bwd_SSIM_pyramid[scale] + (1 - self.alpha) * bwd_recon_pyramid[scale]
            for scale in range(num_scales)
        ]

        return {
            "fwd_full_flow_pyramid": fwd_full_flow_pyramid,
            "bwd_full_flow_pyramid": bwd_full_flow_pyramid,
            "fwd_full_warp_pyramid": fwd_full_warp_pyramid,
            "bwd_full_warp_pyramid": bwd_full_warp_pyramid,
            "fwd_full_error_pyramid": fwd_full_error_pyramid,
            "bwd_full_error_pyramid": bwd_full_error_pyramid,
        }

    def build_losses(self, tgt_tile_pyramid, ref_pyramid, org_output, n, aug_output=None):
        bs = tgt_tile_pyramid[0].shape[0] // n
        num_scales = len(org_output["flow_fwd"][0])
        # NOTE: geometrical consistency
        bwd2fwd_flow_pyramid = [
            self._warp_image(org_output["bwd_full_flow_pyramid"][s],
                             org_output["fwd_full_flow_pyramid"][s])
            for s in range(num_scales)
        ]
        fwd2bwd_flow_pyramid = [
            self._warp_image(org_output["fwd_full_flow_pyramid"][s],
                             org_output["bwd_full_flow_pyramid"][s])
            for s in range(num_scales)
        ]

        fwd_flow_diff_pyramid = [
            torch.abs(bwd2fwd_flow_pyramid[s] +
                      org_output["fwd_full_flow_pyramid"][s])
            for s in range(num_scales)
        ]
        bwd_flow_diff_pyramid = [
            torch.abs(fwd2bwd_flow_pyramid[s] +
                      org_output["bwd_full_flow_pyramid"][s])
            for s in range(num_scales)
        ]

        fwd_consist_bound_pyramid = [
            self.geometric_consistency_beta * org_output["fwd_full_flow_pyramid"][s]
            * 2 ** s for s in range(num_scales)  # WARN: maybe wrong
        ]
        bwd_consist_bound_pyramid = [
            self.geometric_consistency_beta * org_output["bwd_full_flow_pyramid"][s]
            * 2 ** s for s in range(num_scales)
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

        fwd_mask_pyramid = [(fwd_flow_diff_pyramid[s] * 2 ** s <
                             fwd_consist_bound_pyramid[s]).float()
                            for s in range(num_scales)]
        bwd_mask_pyramid = [(bwd_flow_diff_pyramid[s] * 2 ** s <
                             bwd_consist_bound_pyramid[s]).float()
                            for s in range(num_scales)]
        # from IPython import embed
        # from matplotlib import pyplot as plt
        # embed()
        # NOTE: loss
        loss_rigid_warp = 0
        loss_disp_smooth = 0
        loss_full_warp = 0
        loss_full_smooth = 0
        loss_geometric_consistency = 0
        loss_census = 0

        for s in range(num_scales):
            loss_rigid_warp += self.loss_weight_rigid_warp * n / 2 * (
                    torch.mean(org_output["fwd_rigid_error_pyramid"][s])
                    + torch.mean(org_output["bwd_rigid_error_pyramid"][s]))

            loss_disp_smooth += self.loss_weight_disparity_smooth / 2 ** s * self._smooth_loss(
                torch.cat([org_output['disp_tgt'][s][:, 0]] +
                          [org_output['disp_refs'][i][s][:, 0] for i in range(n)]).unsqueeze(1),
                torch.cat((tgt_tile_pyramid[s][:1], ref_pyramid[s]), dim=0))
            loss_full_warp += self.loss_weight_full_warp * n / 2 * (torch.sum(
                torch.mean(org_output["fwd_full_error_pyramid"][s], 1, True) * fwd_mask_pyramid[s]) / torch.mean(
                fwd_mask_pyramid[s]) + torch.sum(
                torch.mean(org_output["bwd_full_error_pyramid"][s], 1, True) * bwd_mask_pyramid[s]) / torch.mean(
                bwd_mask_pyramid[s]))

            loss_full_smooth += self.loss_weight_full_smooth / 2 ** (s + 1) * \
                                (self._flow_smooth_loss(org_output["fwd_full_flow_pyramid"][s], tgt_tile_pyramid[s]) +
                                 self._flow_smooth_loss(org_output["bwd_full_flow_pyramid"][s], ref_pyramid[s]))

            loss_geometric_consistency += self.loss_weight_geometrical_consistency / 2 * (
                    torch.sum(
                        torch.mean(fwd_flow_diff_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                torch.mean(bwd_flow_diff_pyramid[s], 1, True) *
                bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))

            loss_census += (self.loss_weight_census / 2
                            * (self._census_loss(tgt_tile_pyramid[s], org_output["fwd_full_warp_pyramid"][s])
                               + self._census_loss(ref_pyramid[s], org_output["bwd_full_warp_pyramid"][s])))
            print('rigid warp: {} disp smooth: {}'.format(loss_rigid_warp, loss_disp_smooth))
            loss_total = loss_rigid_warp + loss_disp_smooth
            print('full warp: {} full_smooth: {}, geo_con:{}'.format(loss_full_warp, loss_full_smooth,
                                                                     loss_geometric_consistency))
            loss_total += loss_full_warp + \
                          loss_full_smooth + loss_geometric_consistency
            loss_total += loss_census

        aug_loss = None
        if aug_output is not None:
            aug_loss = self._augmentation_loss(aug_output)
        loss_total += aug_loss
        return loss_total

    def _warp_image(self, image, flow):
        b, _, h, w = flow.size()

        flow = flow.permute((0, 2, 3, 1))

        meshy, meshx = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack([torch.stack((meshx, meshy), 2).cuda()] * b)  # [x,y] coordinate
        cor = grid + flow
        cor = cor / torch.tensor([(w - 1) / 2, (h - 1) / 2]).cuda() - 1  # normalization

        warped_ref_img = F.grid_sample(image, cor, padding_mode="zeros")
        valid_points = cor.abs().max(dim=-1)[0] <= 1

        return warped_ref_img

    def _occ_regularization(self, occ):
        zeros_var = torch.zeros_like(occ)
        occ_reg = nn.functional.binary_cross_entropy(occ, zeros_var)

        return occ_reg

    def _reconstruction_loss(self, tgt_img, warped_img):
        diff = tgt_img - (
            warped_img
            # * valid_points.unsqueeze(1).float()
            # * (1 - occ.unsqueeze(1))
            # + tgt_img_scaled * occ
        )

        reconstruction_loss = diff.abs()

        return reconstruction_loss

    def _smooth_loss(self, depth, image):
        gradient_depth_x = gradient_x(depth)  # (TODO)shape: bs,1,h,w
        gradient_depth_y = gradient_y(depth)

        gradient_img_x = gradient_x(image)  # (TODO)shape: bs,3,h,w
        gradient_img_y = gradient_y(image)

        exp_gradient_img_x = torch.exp(-torch.mean(torch.abs(gradient_img_x), 1, True))  # (TODO)shape: bs,1,h,w
        exp_gradient_img_y = torch.exp(-torch.mean(torch.abs(gradient_img_y), 1, True))

        smooth_x = gradient_depth_x * exp_gradient_img_x
        smooth_y = gradient_depth_y * exp_gradient_img_y

        return torch.mean(torch.abs(smooth_x)) + torch.mean(torch.abs(smooth_y))

    def _flow_smooth_loss(self, flow, img):
        # TODO two flows ?= rigid flow + object motion flow
        smoothness = 0
        for i in range(2):
            # TODO shape of flow: bs,channels(2),h,w
            smoothness += self._smooth_loss(flow[:, i, :, :].unsqueeze(1), img)
        return smoothness / 2

    def _augmentation_loss(self, augmentation):
        n = len(augmentation["aug_ref_imgs"])

        loss = 0
        for i in range(n):
            flow_t_pred = augmentation["aug_output"]["flow_bwd"][i][0]
            flow_t = augmentation["psd_gt"]["flow"][i]
            # noc_t = 1 - F.softmax(augmentation["psd_gt"]["occ"][i], dim=1)[:, 1:, :, :]
            # noc_t = 1 - augmentation["psd_gt"]["occ"][i].squeeze(1)

            l_atst = ((flow_t_pred - flow_t).abs() + 0) ** 1
            # l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += l_atst.mean()

        return loss

    def _SSIM_loss(self, x, y, md=1):
        patch_size = 2 * md + 1
        avepooling2d = torch.nn.AvgPool2d(patch_size, stride=1, padding=[1, 1])
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        mu_x = avepooling2d(x)
        mu_y = avepooling2d(y)

        sigma_x = avepooling2d(x ** 2) - mu_x ** 2
        sigma_y = avepooling2d(y ** 2) - mu_y ** 2
        sigma_xy = avepooling2d(x * y) - mu_x * mu_y
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        # L_square = 255**2

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM.permute(0, 2, 3, 1)) / 2, 0, 1)

    def _census_loss(self, tgt_img, warped_img, patch_size=7):
        """Compares the similarity of the census transform of two images."""
        tgt_census_img = census_transform(tgt_img, patch_size)
        warped_census_img = census_transform(warped_img, patch_size)

        hamming = soft_hamming(tgt_census_img, warped_census_img)

        # Set borders of mask to zero to ignore edge effects.
        # padded_mask = zero_mask_border(1 - occ, patch_size).detach()
        diff = abs_robust_loss(hamming)
        # diff *= padded_mask.unsqueeze(1)
        # diff_sum = diff.sum()
        # loss_mean = diff_sum / (padded_mask.sum() + 1e-6)
        return diff.sum().mean()
