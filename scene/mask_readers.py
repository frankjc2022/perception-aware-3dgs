import os
from PIL import Image
from utils.general_utils import PILtoTorch
DEVICE = "cuda"


def load_segment_mask(path, image_name):

    image_path = os.path.join(path, image_name + ".png")
    object_mask = Image.open(image_path)

    orig_w, orig_h = object_mask.size
    scale = 1
    resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_mask = PILtoTorch(object_mask, resolution)
    if resized_mask.shape[0] != 1:
        resized_mask = resized_mask[:1, ...]
    resized_mask[resized_mask > 0] = 1.
    # object_mask = resized_image_rgb[:3, ...]

    object_mask = resized_mask.clamp(0.0, 1.0).to(DEVICE)
    return object_mask


def load_depth_map(path, image_name):

    image_path = os.path.join(path, image_name + ".png")
    depth_mask = Image.open(image_path)

    orig_w, orig_h = depth_mask.size
    scale = 1
    resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_depth_mask = PILtoTorch(depth_mask, resolution)
    if resized_depth_mask.shape[0] != 1:
        resized_depth_mask = resized_depth_mask[:1, ...]
    # resized_depth_mask[resized_depth_mask > 0] = 1.
    # depth_mask = resized_image_rgb[:3, ...]

    # Normalize depth to 0–1 (preserve float32 type)
    resized_depth_mask = resized_depth_mask.float()
    resized_depth_mask = (resized_depth_mask - resized_depth_mask.min()) / (resized_depth_mask.max() - resized_depth_mask.min() + 1e-8)

    # Invert: closer = 1.0, far = 0.0
    # resized_depth_mask = 1.0 - resized_depth_mask

    depth_mask = resized_depth_mask.clamp(0.0, 1.0).to(DEVICE)
    return depth_mask


def load_saliency_map(path, image_name):

    image_path = os.path.join(path, image_name + ".png")
    heatmap = Image.open(image_path).convert("RGB")

    orig_w, orig_h = heatmap.size
    scale = 1
    resolution = (int(orig_w / scale), int(orig_h / scale))

    grayscale = heatmap.convert("L")
    resized_heatmap_tensor = PILtoTorch(grayscale, resolution)

    # normalize
    tensor = resized_heatmap_tensor.float()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    saliency = tensor.clamp(0.0, 1.0).to(DEVICE)
    return saliency


def load_edge_mask(path, image_name):

    image_path = os.path.join(path, image_name + ".png")
    edge_mask = Image.open(image_path).convert("L")

    orig_w, orig_h = edge_mask.size
    scale = 1
    resolution = (int(orig_w / scale), int(orig_h / scale))

    edge_mask_tensor = PILtoTorch(edge_mask, resolution)

    tensor = edge_mask_tensor.float()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    edge_mask = tensor.clamp(0.0, 1.0).to(DEVICE)
    return edge_mask
