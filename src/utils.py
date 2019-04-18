import os
from os.path import join
from PIL import Image

from src.model_config import IMAGE_EXTENSION


def split_test(directory):
    dog_directory = join(directory, "dogs")
    cat_directory = join(directory, "cats")
    for file in os.listdir(directory):
        old_place = join(directory, file)
        if file.startswith("dog."):
            os.rename(old_place, join(dog_directory, file))
        if file.startswith("cat."):
            os.rename(old_place, join(cat_directory, file))


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
