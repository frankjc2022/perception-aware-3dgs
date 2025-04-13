import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# https://github.com/ultralytics/ultralytics
YOLO_MODEL = YOLO('yolo11n-seg.pt')


def generate_segment_mask(image_folder, output_folder, output_mask_folder):

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            results = YOLO_MODEL(image)
            result = results[0] # input one image at a time only

            output_path = os.path.join(output_folder, f"{filename.split('.')[0]}.png")
            result.save(filename=output_path)
            print(f"Saved segment image to: {output_path}")

            binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if result.masks:
                masks = result.masks.data.cpu().numpy()

                for mask in masks:
                    # resize the mask to match the original image size
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    binary_mask = np.maximum(binary_mask, mask_binary)

            output_mask_path = os.path.join(output_mask_folder, f"{filename.split('.')[0]}.png")
            cv2.imwrite(output_mask_path, binary_mask)
            print(f"Saved segment binary mask to: {output_mask_path}")

    print("Segment image and mask processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary segment mask using YOLO model")
    parser.add_argument("-m", "--model_path", required=True, help="Path where the model stored, must contain an images folder.")
    parser.add_argument("-i", "--images", default="images", help="Name of the image folder (default: 'images')")

    args = parser.parse_args()

    image_dir = os.path.join(args.model_path, args.images)
    save_dir = os.path.join(args.model_path, "segment_images")
    save_mask_dir = os.path.join(args.model_path, "segment_masks")
    generate_segment_mask(image_dir, save_dir, save_mask_dir)
