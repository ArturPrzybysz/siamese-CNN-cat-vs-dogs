import numpy as np

from paths import TRAIN_DIR, VALID_DIR
from src.data_generator.triples_creator import random_triples, semi_hard_triples, hard_triples
from src.evaluate_model import evaluate_model
from src.model_config import INPUT_SHAPE, TRAIN_TRIPLES, VALID_TRIPLES, BATCH_SIZE, EPOCHS, MARGIN, EMBEDDING_SIZE
from src.siamese_model import siamese_model
from src.visualisations.plots import plot_accuracy

model = siamese_model(INPUT_SHAPE, encoding_size=EMBEDDING_SIZE)

print(model.summary())

anchors, positives, negatives = random_triples(TRAIN_DIR, triples_count=TRAIN_TRIPLES)

evaluation_triplets = random_triples(VALID_DIR, triples_count=VALID_TRIPLES)
Y_train = np.random.randint(2, size=(1, 2, anchors.shape[0])).T  # Y_train value does not matter

current_best_accuracy = 0
accuracies = []

print("MARGIN", MARGIN)

for epoch in range(EPOCHS):
    history = model.fit([anchors, positives, negatives], Y_train, epochs=20, batch_size=BATCH_SIZE, shuffle=True)
    accuracy = evaluate_model(model, evaluation_triplets)
    if current_best_accuracy < accuracy > 0.6:
        current_best_accuracy = accuracy
        print("Best score: %s" % current_best_accuracy)
        model.save("current_best.h5")

    print("accuracy:", accuracy)

    if epoch < EPOCHS // 2:
        anchors, positives, negatives = semi_hard_triples(TRAIN_DIR, model, triples_count=TRAIN_TRIPLES)
    else:
        if epoch % 2 == 0:
            anchors, positives, negatives = semi_hard_triples(TRAIN_DIR, model, triples_count=TRAIN_TRIPLES)
        else:
            anchors, positives, negatives = hard_triples(TRAIN_DIR, model, triples_count=TRAIN_TRIPLES)

plot_accuracy(accuracies)
