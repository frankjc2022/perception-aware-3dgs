import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate all preprocessing scripts")
    parser.add_argument("-m", "--model_path", required=True, help="Path where the model stored, must contain an images folder.")
    parser.add_argument("-i", "--images", default="images", help="Name of the image folder (default: 'images')")

    args = parser.parse_args()
    model_path = args.model_path
    images_folder = args.images

    os.system(fr"python ..\preprocessing\depth_map.py -m {model_path} -i {images_folder}")
    os.system(fr"python ..\preprocessing\edge_mask.py -m {model_path} -i {images_folder}")
    os.system(fr"python ..\preprocessing\segment_mask.py -m {model_path} -i {images_folder}")
    os.system(fr"python ..\preprocessing\saliency_map.py -m {model_path} -i {images_folder}")