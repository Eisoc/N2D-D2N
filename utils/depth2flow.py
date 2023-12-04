import torch

from utils.inverse_warp import pose_vec2mat


def invert_transform_matrices(transforms):
    # Extract rotation matrices and translation vectors
    R = transforms[:, :3, :3]  # Shape: (B,3,3)
    t = transforms[:, :3, 3:4]  # Shape: (B,3,1)

    # Compute inverse of rotation matrices
    R_inv = torch.transpose(R, 1, 2)  # Shape: (B,3,3)

    # Compute inverse translation
    t_inv = -torch.bmm(R_inv, t)  # Shape: (B,3,1)

    # Combine to get the inverse transforms
    inv_transforms = torch.cat([R_inv, t_inv], dim=2)  # Shape: (B,3,4)

    return inv_transforms


def all_flows_from_depth_pose( org_output, intrinsics, min_depth=1, max_depth=5000):
    disps_tgt = org_output['disp_tgt']
    disps_refs = org_output['disp_refs']
    poses = org_output['pose']
    num_refs = len(disps_refs)
    num_scales = len(disps_tgt)
    flow_refs = []
    # occs = []
    flow_tgt = []
    for i in range(num_refs):
        flow_tgt_one_scale = []
        flow_refs_ond_scale = []
        for j in range(num_scales):
            disp_tgt = disps_tgt[j][:, 0]
            disp_ref = disps_refs[i][j][:, 0]
            pose = poses[:, i]
            downscale = disp_tgt.size(1) / disps_tgt[0][:, 0].size(1)
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)

            _, depth = disp_to_depth(disp_tgt, min_depth, max_depth)
            pose_mat = pose_vec2mat(pose)
            flow_refs_ond_scale.append(optical_flow_from_depth_pose(depth, pose_mat, intrinsics_scaled, False))

            _, depth = disp_to_depth(disp_ref, min_depth, max_depth)
            flow_tgt_one_scale.append(optical_flow_from_depth_pose(depth, pose_mat, intrinsics_scaled, True))
        # occs.append(model._calculate_occlusion(flow_refs_ond_scale, flow_tgt_one_scale))
        flow_refs.append(flow_refs_ond_scale)
        flow_tgt.append(flow_tgt_one_scale)

    org_output["flow_bwd"] = flow_refs  # from tartget to reference
    org_output["flow_fwd"] = flow_tgt
    # org_output["occ"] = occs


def optical_flow_from_depth_pose(depth, pose, intrinsics, reverse_pose):
    """Calculate optical flow from depth and pose.
    """
    B, H, W = depth.shape

    filler = torch.tensor([0., 0., 0., 1.]).view(1, 4).repeat(B, 1, 1).float().to("cuda")
    homo_pose = torch.cat((pose, filler), dim=1)
    if reverse_pose:
        homo_pose = torch.inverse(homo_pose)  # (b, 4, 4)

    pixel_coords = meshgrid(B, H, W)  # (batch, 3, h, w)

    tgt_pixel_coords = pixel_coords[:, :2, :, :].permute(0, 2, 3, 1)  # (batch, h, w, 2)
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)  # (batch, 4, h, w)

    # Construct 4x4 intrinsics matrix
    filler = torch.tensor([0., 0., 0., 1.]).view(1, 4).repeat(B, 1, 1).to("cuda")
    intrinsics = torch.cat((intrinsics, torch.zeros((B, 3, 1)).float().to("cuda")), dim=2)
    intrinsics = torch.cat((intrinsics, filler), dim=1)  # (batch, 4, 4)

    proj_tgt_cam_to_src_pixel = torch.matmul(intrinsics, homo_pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)

    rigid_flow = src_pixel_coords - tgt_pixel_coords

    return rigid_flow.permute(0, 3, 1, 2)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """

    b, h, w = depth.size()

    depth = depth.view(b, 1, -1)
    pixel_coords = pixel_coords.view(b, 3, -1)
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth

    if is_homogeneous:
        ones = torch.ones(b, 1, h * w).float().to("cuda")
        cam_coords = torch.cat((cam_coords.to("cuda"), ones), dim=1)

    cam_coords = cam_coords.view(b, -1, h, w)

    return cam_coords


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    b, _, h, w = cam_coords.size()
    cam_coords = cam_coords.view(b, 4, h * w)
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)

    x_u = unnormalized_pixel_coords[:, :1, :]
    y_u = unnormalized_pixel_coords[:, 1:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]

    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)

    pixel_coords = torch.cat((x_n, y_n), dim=1)
    pixel_coords = pixel_coords.view(b, 2, h, w)

    return pixel_coords.permute(0, 2, 3, 1)


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates

    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """

    # (height, width)
    x_t = torch.matmul(
        torch.ones(height).view(height, 1).float().to("cuda"),
        torch.linspace(-1, 1, width).view(1, width).to("cuda"))

    # (height, width)
    y_t = torch.matmul(
        torch.linspace(-1, 1, height).view(height, 1).to("cuda"),
        torch.ones(width).view(1, width).float().to("cuda"))

    x_t = (x_t + 1) * 0.5 * (width - 1)
    y_t = (y_t + 1) * 0.5 * (height - 1)

    if is_homogeneous:
        ones = torch.ones_like(x_t).float().to("cuda")
        # ones = torch.ones(height, width).float().to("cuda")
        coords = torch.stack((x_t, y_t, ones), dim=0)  # shape: 3, h, w
    else:
        coords = torch.stack((x_t, y_t), dim=0)  # shape: 2, h, w

    coords = torch.unsqueeze(coords, 0).expand(batch, -1, height, width)

    return coords
