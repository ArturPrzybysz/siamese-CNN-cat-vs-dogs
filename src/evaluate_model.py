from src.utils import prediction_to_positive_dist, prediction_to_negative_dist


def evaluate_model(model, evaluated_triples):
    anchors, positives, negatives = evaluated_triples
    predictions = model.predict([anchors, positives, negatives])
    correct = 0
    _all = 0
    for prediction in predictions:
        positive_dist = prediction_to_positive_dist(prediction)
        negative_dist = prediction_to_negative_dist(prediction)
        _all += 1
        if positive_dist < negative_dist:
            correct += 1

    return correct / _all
