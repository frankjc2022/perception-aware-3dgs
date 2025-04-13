#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    ############## extra masks ############
    depth_map = cam_info.depth_map
    edge_mask = cam_info.edge_mask
    saliency_map = cam_info.saliency_map
    segment_mask = cam_info.segment_mask
    #######################################

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, depth_map=depth_map, edge_mask=edge_mask, saliency_map=saliency_map, segment_mask=segment_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def generate_unseen_cameras(FoVx, FoVy, image_width, image_height, center, radius, num_cameras):

    cameras = []
    center_np = center

    for idx in range(num_cameras):
        angle = idx * 2. * np.pi / num_cameras
        x = np.cos(angle) * radius
        y = 0
        z = np.sin(angle) * radius

        campos = np.array([x, y, z], dtype=np.float32)

        lookat = center_np
        up = np.array([0., -1., 0.], dtype=np.float32)
        forward = lookat - campos
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        camrot = np.array([right, -up, forward], dtype=np.float32)

        camT = -camrot @ campos

        R_torch = torch.tensor(camrot, dtype=torch.float32, device="cuda")
        T_torch = torch.tensor(camT, dtype=torch.float32, device="cuda")

        camera = Camera(
            colmap_id=idx,
            R=R_torch.cpu().numpy(),
            T=T_torch.cpu().numpy(),
            FoVx=FoVx,
            FoVy=FoVy,
            image=None,
            gt_alpha_mask=None,
            image_name="",
            uid="",
            data_device="cuda",
            image_size=(image_height, image_width)
        )
        cameras.append(camera)

    return cameras