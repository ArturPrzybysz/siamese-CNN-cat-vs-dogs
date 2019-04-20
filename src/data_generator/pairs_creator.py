from src.data_generator.data_gen_utils import read_random_images_from_directory
from os.path import join
import numpy as np


def create_evaluation_pairs(data_directory: str, count: int):
    dogs = read_random_images_from_directory(join(data_directory, "dogs"), count)
    cats = read_random_images_from_directory(join(data_directory, "cats"), count)

    assert len(dogs) == len(cats) and len(dogs) > 0

    return np.array(dogs), np.array(cats)
