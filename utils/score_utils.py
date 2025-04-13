import torch
from scene.mask_readers import load_segment_mask, load_depth_map, load_saliency_map, load_edge_mask


def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value


def compute_gaussian_score_with_perception(scene, camlist, edge_losses, gaussians, pipe, bg, importance_values, opt):
    """
    All masks are loaded here when needed. If we preload in camera, then it will be so slow to run.
    Since we only run it like every 500 iteration, load it only need it, it's much faster.
    """

    importance_values = {
        "depth_importance": 50,
        "saliency_importance": 100,
        "edge_importance": 25,
        "segment_importance": 50
    }

    num_points = len(scene.gaussians.get_xyz)
    gaussian_importance = torch.zeros((len(camlist), num_points), device="cuda", dtype=torch.float32)

    for view in range(len(camlist)):

        my_viewpoint_cam = camlist[view]

        depth_scores = 0
        saliency_scores = 0
        edge_scores = 0
        segment_scores = 0

        if my_viewpoint_cam.depth_map:
            depth_scores = means3D_score(my_viewpoint_cam, scene.gaussians.get_xyz, load_depth_map(my_viewpoint_cam.depth_map, my_viewpoint_cam.image_name))
            depth_scores = normalize(importance_values["depth_importance"], depth_scores)
        if my_viewpoint_cam.saliency_map:
            saliency_scores = means3D_score(my_viewpoint_cam, scene.gaussians.get_xyz, load_saliency_map(my_viewpoint_cam.saliency_map, my_viewpoint_cam.image_name))
            saliency_scores = normalize(importance_values["saliency_importance"], saliency_scores)
        if my_viewpoint_cam.edge_mask:
            edge_scores = means3D_score(my_viewpoint_cam, scene.gaussians.get_xyz, load_edge_mask(my_viewpoint_cam.edge_mask, my_viewpoint_cam.image_name))
            edge_scores = normalize(importance_values["edge_importance"], edge_scores)
        if my_viewpoint_cam.segment_mask:
            segment_scores = means3D_score(my_viewpoint_cam, scene.gaussians.get_xyz, load_segment_mask(my_viewpoint_cam.segment_mask, my_viewpoint_cam.image_name))
            segment_scores = normalize(importance_values["segment_importance"], segment_scores)

        total_score = (
                depth_scores +
                saliency_scores +
                edge_scores +
                segment_scores
        )

        gaussian_importance[view] = total_score

    gaussian_importance = gaussian_importance.sum(dim=0)

    return gaussian_importance


def means3D_score(viewpoint_camera, points3D, mask):

    device = points3D.device
    mask = mask.to(device)

    # convert the shape of the mask
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    N = points3D.shape[0]
    h, w = viewpoint_camera.image_height, viewpoint_camera.image_width

    # add homogeneous coordinate to points
    # ones = torch.ones((N, 1), device=points3D.device, dtype=points3D.dtype)
    # print(f"ones: {ones.shape}")
    # print(f"points3D: {points3D.shape}")
    # points_homo = torch.cat([points3D, ones], dim=1).to("cuda")  # [N, 4]

    # points3D = points3D.to("cuda")
    # ones = torch.ones((points3D.shape[0], 1), device="cuda", dtype=points3D.dtype)
    # print(f"ones: {ones.shape}")
    # print(f"points3D: {points3D.shape}")
    # points_homo = torch.cat([points3D, ones], dim=1).to("cuda")

    points3D = points3D.to("cuda")
    ones = torch.ones((points3D.shape[0], 1), device=points3D.device, dtype=points3D.dtype)
    points_homo = torch.cat([points3D, ones], dim=1)

    # apply camera's projection transform, project to image plane (world to camera @ intrinsic K, check cameras.py for calculation)
    points_homo = (points_homo @ viewpoint_camera.full_proj_transform).transpose(0, 1)

    # normalized by the w (the last w for normalize xyz), to get normalized device coordinates NDC
    points_w = 1.0 / (points_homo[-1, :] + 1e-7)
    points_proj = points_homo[:3, :] * points_w

    # NDC to pixel coordinates
    # https://stackoverflow.com/questions/4578786/normalized-device-coordinates-to-window-coordinates
    pixels = 0.5 * ((points_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(points_proj.device) - 1)
    pixels = torch.round(pixels.transpose(0, 1))

    x = pixels[:, 0].long()
    y = pixels[:, 1].long()

    # make sure it's within the image size
    clamped_x = torch.clamp(x, min=0, max=w - 1)
    clamped_y = torch.clamp(y, min=0, max=h - 1)

    scores = mask[clamped_y, clamped_x]

    return scores