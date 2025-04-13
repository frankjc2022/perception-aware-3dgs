import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import sys
import argparse
sys.path.append(r'./U-2-Net')
from model import U2NETP

U2NET_WEIGHT_PATH = r'./u2netp.pth'

def load_u2net_model():
    model = U2NETP(3, 1)
    model.load_state_dict(torch.load(U2NET_WEIGHT_PATH, map_location='cpu'))
    model.eval()
    return model


def get_saliency_u2net(image_path, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = model(tensor)
        saliency = d1.squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # resize
    saliency_tensor = torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0)
    saliency_resized = F.interpolate(saliency_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    return saliency_resized.squeeze().numpy()  # shape [H, W], float32 in 0–1


def generate_saliency_map(image_folder, output_folder):

    model = load_u2net_model()

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            saliency = get_saliency_u2net(image_path, model)

            # convert to 8-bit grayscale heat map
            saliency_uint8 = (saliency * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)

            # save image
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(output_path, heatmap)
            print(f"Saliency map saved to: {output_path}")

    print("Saliency map processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate saliency map using u2net.")
    parser.add_argument("-m", "--model_path", required=True, help="Path where the model stored, must contain an images folder.")
    parser.add_argument("-i", "--images", default="images", help="Name of the image folder (default: 'images')")

    args = parser.parse_args()

    image_dir = os.path.join(args.model_path, args.images)
    save_dir = os.path.join(args.model_path, "saliency_maps")
    generate_saliency_map(image_dir, save_dir)
