import numpy as np

from paths import TRAIN_DIR, VALID_DIR
from src.data_generator.pairs_creator import create_evaluation_pairs
from src.data_generator.triples_creator import random_triples, hard_triples
from src.evaluate_model import evaluate_model
from src.model_config import INPUT_SHAPE, TRAIN_TRIPLES, VALID_PAIRS, BATCH_SIZE, EPOCHS, MARGIN
from src.siamese_model import siamese_model
from src.visualisations.plots import plot_dist_var

model = siamese_model(INPUT_SHAPE, encoding_size=4)

print(model.summary())

anchors, positives, negatives = random_triples(TRAIN_DIR, triples_count=TRAIN_TRIPLES)

evaluation_pairs = create_evaluation_pairs(VALID_DIR, count=VALID_PAIRS)
Y_train = np.random.randint(2, size=(1, 3, anchors.shape[0])).T  # Y_train value does not matter
cats_vs_dogs_distances = []
dogs_vs_dogs_distances = []
cats_vs_cats_distances = []

for epoch in range(EPOCHS):
    history = model.fit([anchors, positives, negatives], Y_train, epochs=2, batch_size=BATCH_SIZE)
    cd_dist, dd_dist, cc_dist = evaluate_model(model, evaluation_pairs)
    cats_vs_dogs_distances.append(cd_dist)
    dogs_vs_dogs_distances.append(dd_dist)
    cats_vs_cats_distances.append(cc_dist)

    anchors, positives, negatives, high_margin = hard_triples(TRAIN_DIR, model, triples_count=TRAIN_TRIPLES)
    if high_margin and MARGIN > 0.01:
        MARGIN -= 0.01
    else:
        MARGIN += 0.05
    print(MARGIN, cd_dist, dd_dist, cc_dist)

print("Dogs vs dogs:", cats_vs_dogs_distances)
print("Dogs vs dogs:", dogs_vs_dogs_distances)
print("Dogs vs dogs:", cats_vs_cats_distances)
plot_dist_var(cats_vs_dogs_distances, dogs_vs_dogs_distances, cats_vs_cats_distances)
