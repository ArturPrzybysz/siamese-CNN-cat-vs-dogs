import os
from PIL import Image

from src.model_config import IMAGE_EXTENSION


def show_images(images):
    for image in images:
        img = Image.fromarray(image, 'RGB')
        img.show()


def count_images_in_directory(dir_path: str):
    files = os.listdir(dir_path)
    count = 0
    for f in files:
        if f.endswith(IMAGE_EXTENSION):
            count += 1
    return count
