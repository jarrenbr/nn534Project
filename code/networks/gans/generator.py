import tensorflow as tf

from networks import defaults
from files import filePaths as fp

BATCH_SIZE = 16


def get_gen_input(shape=(defaults.BATCH_SIZE, defaults.NOISE_DIM), minMax = defaults.MIN_MAX_RNG):
    data = tf.random.uniform(shape=shape, minval=minMax[0], maxval=minMax[1])
    return data

