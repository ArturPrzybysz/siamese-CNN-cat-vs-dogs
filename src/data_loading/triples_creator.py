import numpy as np
import cv2
from os.path import join
import os
from random import shuffle, sample

from src.model_config import IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, IMAGE_EXTENSION


def random_triples(data_dir: str, triples_count=0):
    dog_directory = join(data_dir, "dogs")
    cat_directory = join(data_dir, "cats")

    dog_images = read_random_images_from_directory(dog_directory, triples_count)
    cat_images = read_random_images_from_directory(cat_directory, triples_count)

    anchor_dogs = sample(dog_images, triples_count // 2)
    anchor_cats = sample(cat_images, triples_count - triples_count // 2)

    positive_dogs = sample(dog_images, len(anchor_dogs))
    positive_cats = sample(cat_images, len(anchor_cats))

    negative_cats = sample(cat_images, len(anchor_dogs))
    negative_dogs = sample(dog_images, len(anchor_cats))

    anchor = np.array(anchor_dogs + anchor_cats)
    positive = np.array(positive_dogs + positive_cats)
    negative = np.array(negative_cats + negative_dogs)

    assert anchor[0].shape == (IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)

    return anchor, positive, negative


def triples_by_model(data_dir: str, model, triples_count=0):
    # TODO
    pass


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
            images.append(_read_image(join(directory, file)))

    return images
