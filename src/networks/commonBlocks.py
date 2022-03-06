import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l

from networks import defaults

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l

from networks import defaults

class conv_args:
    def __init__(self, nFilters, kernelSize, strides=2, padding="valid", useBias=True):
        self.nFilters = nFilters
        self.kernelSize = kernelSize
        self.strides = strides
        self.padding = padding
        self.useBias = useBias

        self.kwargs = {"filters": self.nFilters, "kernel_size" : self.kernelSize,
                       "strides": strides, "padding" : self.padding, "use_bias" : self.useBias}


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
    x = l.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    x = block(
        x,
        activation=activation,
        use_bn=use_bn, use_dropout=use_dropout, drop_value=drop_value,
    )
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