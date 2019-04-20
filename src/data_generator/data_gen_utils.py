from random import shuffle

import cv2
import os

from src.model_config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_EXTENSION


def _read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return t_image


def read_random_images_from_directory(directory: str, image_count: int):
    images = []
    files = os.listdir(directory)
    shuffle(files)

    for i in range(image_count):
        file = files[i]
        if file.endswith(IMAGE_EXTENSION):
            images.append(_read_image(os.path.join(directory, file)))

    return images
