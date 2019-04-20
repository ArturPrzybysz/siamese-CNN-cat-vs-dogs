import numpy as np


def evaluate_model(model, evaluated_pairs):
    dogs, cats = evaluated_pairs
    dummy_anchors = np.zeros_like(dogs)

    response = model.predict([dummy_anchors, dogs, cats])
    dogs_vs_cats_dist = response[:, 2, :]

    response = model.predict([dummy_anchors, dogs, np.flip(dogs)])
    dogs_vs_dogs_dist = response[:, 2, :]

    response = model.predict([dummy_anchors, cats, np.flip(cats)])
    cats_vs_cats_dist = response[:, 2, :]

    return np.mean(dogs_vs_cats_dist), np.mean(dogs_vs_dogs_dist), np.mean(cats_vs_cats_dist)
