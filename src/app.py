import numpy as np

from paths import TRAIN_DIR
from src.data_loading.triples_creator import random_triples
from src.model_config import INPUT_SHAPE, BATCH_SIZE
from src.siamese_model import get_siamese_model

model = get_siamese_model(INPUT_SHAPE, 15)

epochs = 10

for epoch in range(10):
    anchors, positives, negatives = random_triples(TRAIN_DIR, 500)
    Y_train = np.random.randint(2, size=(1, 3, anchors.shape[0])).T  # Y_train value does not matter

    model.fit([anchors, positives, negatives], Y_train, epochs=10, batch_size=BATCH_SIZE)
