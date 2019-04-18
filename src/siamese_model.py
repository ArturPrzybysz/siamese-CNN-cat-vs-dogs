from tensorflow.python.keras import Input, Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import glorot_normal


def _triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - 0.5 * (
            K.square(y_pred[:, 1, 0]) + K.square(y_pred[:, 2, 0])) + margin))


def get_siamese_model(input_shape, encoding_size):
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    model = Sequential()

    model.add(Conv2D(32, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=glorot_normal(), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())

    model.add(Conv2D(48, (7, 7), activation='relu', kernel_initializer=glorot_normal()))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (4, 4), activation='relu', kernel_initializer=glorot_normal()))
    model.add(MaxPooling2D())

    model.add(Conv2D(82, (3, 3), activation='relu', kernel_initializer=glorot_normal()))
    model.add(Flatten())

    model.add(Dense(512, activation='sigmoid', kernel_initializer=glorot_normal()))

    model.add(Dense(encoding_size, activation='relu', name='embedding'))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1), name='t_emb_1_l2norm'))

    encoded_anchor = model(anchor_input)
    encoded_positive = model(positive_input)
    encoded_negative = model(negative_input)

    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))

    positive_dist = L1_layer([encoded_anchor, encoded_positive])
    negative_dist = L1_layer([encoded_anchor, encoded_negative])
    tertiary_dist = L1_layer([positive_dist, negative_dist])

    stacked_dists = Lambda(lambda v: K.stack(v, axis=1), name='stacked_dists')(
        [positive_dist, negative_dist, tertiary_dist])

    model = Model([anchor_input, positive_input, negative_input], stacked_dists, name='triple_siamese')
    model.compile(optimizer=Adam(), loss=_triplet_loss, metrics=["accuracy"])

    return model
