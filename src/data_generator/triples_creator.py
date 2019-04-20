import numpy as np
from os.path import join
from random import sample

from src.data_generator.data_gen_utils import read_random_images_from_directory
from src.model_config import IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, MARGIN

from tensorflow.python.keras import Model


def hard_triples(data_dir: str, model: Model, triples_count: int):
    dog_dir = join(data_dir, "dogs")
    cat_dir = join(data_dir, "cats")

    imgs_per_iteration_per_class = triples_count

    anchors = []
    positives = []
    negatives = []

    for i in np.arange(2):
        dog_anchor_images = read_random_images_from_directory(dog_dir, imgs_per_iteration_per_class)
        cat_anchor_images = read_random_images_from_directory(cat_dir, imgs_per_iteration_per_class)

        dog_positives = read_random_images_from_directory(dog_dir, imgs_per_iteration_per_class)
        cat_positives = read_random_images_from_directory(cat_dir, imgs_per_iteration_per_class)

        cat_negatives = read_random_images_from_directory(cat_dir, imgs_per_iteration_per_class)
        dog_negatives = read_random_images_from_directory(dog_dir, imgs_per_iteration_per_class)

        tmp_anchors = dog_anchor_images + cat_anchor_images
        tmp_positives = dog_positives + cat_positives
        tmp_negatives = cat_negatives + dog_negatives

        predictions = model.predict([np.array(tmp_anchors), np.array(tmp_positives), np.array(tmp_negatives)])
        for j in np.arange(len(predictions)):
            positive_dist = predictions[j, 0, 0]
            negative_dist = predictions[j, 1, 0]

            if positive_dist + MARGIN < negative_dist:
                anchors.append(tmp_anchors[i])
                positives.append(tmp_positives[i])
                negatives.append(tmp_negatives[i])

                if len(anchors) == triples_count:
                    assert len(anchors) == len(positives) == len(negatives)
                    high_margin = False
                    return np.array(anchors), np.array(positives), np.array(negatives), high_margin

    print("Triples had to be filled with %d random triples out of %d" % (triples_count - len(anchors), triples_count))
    random_anchors, random_positive, random_negative = random_triples(data_dir, triples_count - len(anchors),
                                                                      to_numpy=False)
    anchors = np.array(anchors + random_anchors)
    positives = np.array(positives + random_positive)
    negatives = np.array(negatives + random_negative)
    high_margin = True
    return anchors, positives, negatives, high_margin


def _choose_directories(dirs, i):
    dir1, dir2 = dirs
    if i % 2 == 0:
        return dir1, dir2
    else:
        return dir2, dir1


def random_triples(data_dir: str, triples_count, to_numpy=True):
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

    anchors = anchor_dogs + anchor_cats
    positive = positive_dogs + positive_cats
    negative = negative_cats + negative_dogs

    if to_numpy:
        anchors = np.array(anchors)
        positive = np.array(positive)
        negative = np.array(negative)

    assert anchors[0].shape == (IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)
    assert len(anchors) == len(positive) == len(negative)

    return anchors, positive, negative
