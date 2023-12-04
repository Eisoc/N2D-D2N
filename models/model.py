import torch
import torch.nn as nn

from .FlowNet import FlowNet
from .common import weight_init
from .monodepth2 import DepthDecoder
from .posenet import PoseExpNet
from .psmnet import PSMEncoder
from .pwc import PWCFlowDecoder


class Net(nn.Module):
    def __init__(self, config, compose=None, flow_compose=None, occ_compose=None):
        super().__init__()
        # self.compose = compose
        # self.flow_compose = flow_compose
        # self.occ_compose = occ_compose
        # self.stages = config["model"]["stages"]

        self.encoder = PSMEncoder(with_ppm=config["with_ppm"])
        enc_ch_num = [32, 32, 64, 128, 128]
        # self.encoder = PWCEncoder(with_ppm=config['with_ppm)
        # enc_ch_num = [16, 32, 64, 96, 128]

        self.do_depth = config["depth_enabled"]
        self.do_flow = config["flow_enabled"]

        if self.do_flow:
            self.flow_decoder = PWCFlowDecoder(
                encoder_planes=enc_ch_num,
                md=config["corr_radius"],
                refinement_module=config["flow_refinement"],
                pred_occ=not config["no_occ"],
                cat_occ=config["cat_occ"],
                upsample_output=config["upsample_flow_output"],
            )
        else:
            self.flow_decoder = None

        if self.do_depth:
            self.depth_decoder = DepthDecoder(enc_ch_num, num_output_channels=2)

            self.pose_net = PoseExpNet(2)
        else:
            self.depth_decoder = None
            self.pose_net = None

        self.flow_net = FlowNet(input_chnls=12, flow_scale_factor=config["flow_scale_factor"])

        weight_init(self)

    def forward(self, tgt_img, ref_imgs):
        output = {}
        if not isinstance(ref_imgs, list):
            ref_imgs = [ref_imgs]
            raise Exception("reference should be list type")

        tgt_feat = self.encoder(tgt_img)

        # disp and pose net
        if self.do_depth:
            disp = self.depth_decoder(tgt_feat)
            disp_refs = [self.depth_decoder(self.encoder(ref_img)) for ref_img in ref_imgs]
            pose = self.pose_net(tgt_img, ref_imgs)
        else:
            disp = None
            disp_refs = None
            pose = None

        # the seconde stage is used for augmentation
        if self.do_flow:
            flow_refs, occ_refs = [], []
            flow_tgt = []
            for ref_img in ref_imgs:
                ref_feat = self.encoder(ref_img)
                flow, _ = self.flow_decoder(tgt_feat, ref_feat)
                flow_refs.append(flow)  # tgt to refs
                flow_t, _ = self.flow_decoder(ref_feat, tgt_feat)
                flow_tgt.append(flow_t)  # refs to tgt
                occ = self._calculate_occlusion(flow, flow_t)
                occ_refs.append(occ)
        else:
            flow_refs = None
            occ_refs = None
            flow_tgt = None

        return {
            "aug_tgt": tgt_img,
            "aug_refs": ref_imgs,
            "disp": disp,
            "disp_refs": disp_refs,
            "pose": pose,
            "flow": flow_refs,
            "occ": occ_refs,
            "flow_tgt": flow_tgt,
            # "flow": [flow_ref[0] for flow_ref in flow_refs],
            # "occ": [occ_ref[0] for occ_ref in occ_refs],
        }

    def _calculate_occlusion(self,forward_flow, backward_flow, threshold=1.0):
        """ Calculate the occlusion tensor using forward and backward optical flows.

        Parameters:
        - forward_flow: Pytorch tensor of shape [B, 2, H, W] representing the forward optical flow for each image in the batch.
        - backward_flow: Pytorch tensor of shape [B, 2, H, W] representing the backward optical flow for each image in the batch.
        - threshold: Pixels with displacement values above this threshold will be considered occluded.

        Returns:
        - Occlusion tensor of shape [B, H, W].
        """
        if isinstance(forward_flow, tuple):
            occlusion_tensors = []
            for foward_flow_i, backward_flow_i in zip(forward_flow, backward_flow):
                occlusion_tensors.append(self._calculate_occlusion(foward_flow_i, backward_flow_i, threshold))
            return occlusion_tensors

        B, _, H, W = forward_flow.shape
        occlusion_tensors = []

        for b in range(B):
            # Create a grid of the same shape as the flow for each image in the batch.
            grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            grid = torch.stack((grid_x, grid_y), 0).float().cuda()  # Shape: [2, H, W]

            # Map pixels of the source image to the destination image using the forward flow.
            mapped_fwd = grid + forward_flow[b]

            # Use backward flow to map back to source.
            h_coords = torch.clamp(mapped_fwd[1].long(), 0, H - 1)
            w_coords = torch.clamp(mapped_fwd[0].long(), 0, W - 1)

            # Get backward flow values at the mapped coordinates.
            backward_at_mapped = backward_flow[b, :, h_coords, w_coords]

            # Map pixels of the mapped positions back to the source image.
            mapped_back = mapped_fwd + backward_at_mapped

            # Calculate the difference between the original positions and the mapped-back positions.
            diff = torch.norm(grid - mapped_back, dim=0)

            # Append to the list of occlusion tensors for each image in the batch.
            occlusion_tensors.append((diff > threshold).float().cuda())

        # Stack occlusion tensors to get the final output shape [B, 1, H, W]
        occlusion_batched = torch.stack(occlusion_tensors, dim=0).unsqueeze(1)

        return occlusion_batched

    def pose_and_disp(self, tgt_img, ref_imgs):
        # images = torch.cat([tgt_img] + ref_imgs, 0)  # (bs+bs*ref, 3 , h, w)
        # features = self.encoder(images)
        # disp = self.depth_decoder(features)
        tgt_feat = self.encoder(tgt_img)
        disp = self.depth_decoder(tgt_feat)
        disp_refs = [self.depth_decoder(self.encoder(ref_img)) for ref_img in ref_imgs]
        pose = self.pose_net(tgt_img, ref_imgs)
        return {"disp_tgt": disp, "disp_refs": disp_refs, "pose": pose}
