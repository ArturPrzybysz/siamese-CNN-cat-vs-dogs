from tensorflow.python.keras import Input, Sequential, Model
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.applications import MobileNet

from src.model_config import MARGIN, LR


def _triplet_loss(_, y_pred):
    margin = K.constant(MARGIN)
    positive_dist = y_pred[:, 0, 0]
    negative_dist = y_pred[:, 1, 0]

    basic_loss = K.square(positive_dist) - K.square(negative_dist) + margin

    return K.mean(K.maximum(K.constant(0), basic_loss))


def siamese_model(input_shape, encoding_size):
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    mobile = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet', alpha=0.25)

    model = Sequential()
    model.add(mobile)

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid', kernel_initializer=glorot_normal()))
    model.add(Dropout(0.3))

    model.add(Dense(encoding_size, activation='relu', name='embedding'))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm'))

    encoded_anchor = model(anchor_input)
    encoded_positive = model(positive_input)
    encoded_negative = model(negative_input)

    L2_dist = Lambda(_euclidean_distance, name='L2_dist')
    positive_dist = L2_dist([encoded_anchor, encoded_positive])
    negative_dist = L2_dist([encoded_anchor, encoded_negative])

    stacked_dists = Lambda(lambda v: K.stack(v, axis=1), name='stacked_dists') \
        ([positive_dist, negative_dist])

    model = Model([anchor_input, positive_input, negative_input], stacked_dists, name='siamese')
    model.compile(optimizer=Adam(lr=LR, decay=0.000000065), loss=_triplet_loss)

    return model


def _euclidean_distance(tensors):
    x, y = tensors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
