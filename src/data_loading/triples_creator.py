import numpy as np
import cv2
from os.path import join

from src.model_config import IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS


def random_triples(data_dir: str, triples_count=0):
    cats = read_img_from_directory(join(data_dir, "cats"), triples_count)
    dogs = read_img_from_directory(join(data_dir, "cats"), triples_count)

    dog_count = triples_count // 2
    cat_count = triples_count - dog_count

    anchor_dog_idx =



def triples_by_model(data_dir: str, model, triples_count=0):
    pass


def _read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    t_image = t_image.astype("float32")
    return t_image


def read_img_from_directory(directory: str, number: int):

