import os
import cv2
import numpy as np
import argparse

def generate_edge_mask(image_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Sobel
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.sqrt(sobelx ** 2 + sobely ** 2)

            # Normalize to 0–255
            edge_norm = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min() + 1e-8)
            edge_uint8 = (edge_norm * 255).astype(np.uint8)

            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(output_path, edge_uint8)
            print(f"Edge mask saved to: {output_path}")

    print("Edge mask processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edge mask using sobel operator.")
    parser.add_argument("-m", "--model_path", required=True, help="Path where the model stored, must contain an images folder.")
    parser.add_argument("-i", "--images", default="images", help="Name of the image folder (default: 'images')")

    args = parser.parse_args()

    image_dir = os.path.join(args.model_path, args.images)
    save_dir = os.path.join(args.model_path, "edge_masks")
    generate_edge_mask(image_dir, save_dir)
