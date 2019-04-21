from paths import TRAIN_DIR
from src.data_generator.triples_creator import hard_triples
from src.model_config import TRAIN_TRIPLES, INPUT_SHAPE, EMBEDDING_SIZE
from src.siamese_model import siamese_model

model = siamese_model(INPUT_SHAPE, encoding_size=EMBEDDING_SIZE)

anchors, positives, negatives = hard_triples(TRAIN_DIR, model, triples_count=TRAIN_TRIPLES)

