import numpy as np


def evaluate_model(model, evaluated_triples):
    anchors, positives, negatives = evaluated_triples
    predictions = model.predict([anchors, positives, negatives])
    return np.mean(predictions[:, 0, 0] < predictions[:, 1, 0])
