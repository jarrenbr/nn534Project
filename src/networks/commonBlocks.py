import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l

from networks import defaults


def block(
    x,
    activation,
    use_bn=False,
    use_dropout=False,
    drop_value = defaults.DROPOUT_PORTION
):
    if use_bn:
        x = l.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    return x

