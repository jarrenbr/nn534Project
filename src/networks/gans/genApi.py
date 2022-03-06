import tensorflow as tf

from networks import defaults

def get_gen_input(shape=(defaults.BATCH_SIZE, defaults.NOISE_DIM), minMax = defaults.MIN_MAX_RNG):
    data = tf.random.uniform(shape=shape, minval=minMax[0], maxval=minMax[1])
    return data

def gen_input(shape=(defaults.BATCH_SIZE, defaults.NOISE_DIM), minMax = defaults.MIN_MAX_RNG):
    while True:
        yield get_gen_input(shape=(defaults.BATCH_SIZE, defaults.NOISE_DIM), minMax = defaults.MIN_MAX_RNG)

def get_gen_out(gen, noiseDim=defaults.NOISE_DIM, batchSize=defaults.BATCH_SIZE, training=False):
    return gen(get_gen_input((batchSize, noiseDim)), training=training)
