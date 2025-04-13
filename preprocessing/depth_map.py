import os
import torch
from PIL import Image
import numpy as np
import matplotlib
import argparse

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
repo = "isl-org/ZoeDepth"
# https://github.com/isl-org/ZoeDepth
# ZoeD_N, ZoeD_K, ZoeD_NK
# ZoeD_N for indoor scenes, ZoeD_K for outdoor road scenes, and ZoeD_NK for generic scenes
# https://github.com/isl-org/ZoeDepth/issues/10
model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ZOEDEPTH_MODEL = model_zoe_n.to(DEVICE)


# for colorize the depth map, code from: https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/utils/misc.py
def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def generate_depth_map(image_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")

            depth_pil = ZOEDEPTH_MODEL.infer_pil(image)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")

            colored = colorize(depth_pil)
            Image.fromarray(colored).save(output_path)

            print(f"Saved depth map to: {output_path}")

    print("Depth map processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate depth map using ZoeDepth")
    parser.add_argument("-m", "--model_path", required=True, help="Path where the model stored, must contain an images folder.")
    parser.add_argument("-i", "--images", default="images", help="Name of the image folder (default: 'images')")

    args = parser.parse_args()

    image_dir = os.path.join(args.model_path, args.images)
    save_dir = os.path.join(args.model_path, "depth_maps")
    generate_depth_map(image_dir, save_dir)
