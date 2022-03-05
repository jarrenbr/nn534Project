import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l

from networks import defaults

def conv_block(
    x,
    filters,
    activation,
    kernel_size=3,
    strides=2,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=defaults.DROPOUT_PORTION,
):
    if use_bn:
        use_bias = False
    x = l.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = l.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    return x

def trans_conv_block(
    x,
    filters,
    activation,
    kernel_size=3,
    strides=2,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=defaults.DROPOUT_PORTION,
):


    return x