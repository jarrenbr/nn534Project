import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def upsample_block(
    x,
    filters,
    activation,
    kernel_size=2,
    strides=1,
    up_size=2,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=gv.DROPOUT_DEFAULT,
):
    if use_bn:
        use_bias=False

    if up_size > 1: x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def conv_block():
    pass