import numpy as np
import tensorflow as tf

from networks import defaults

def get_gen_input(shape=(defaults.BATCH_SIZE, defaults.NOISE_DIM), minMax = defaults.MIN_MAX_RNG):
    data = tf.random.uniform(shape=shape, minval=minMax[0], maxval=minMax[1])
    return data

def gen_input(shape=(defaults.BATCH_SIZE, defaults.NOISE_DIM), minMax = defaults.MIN_MAX_RNG):
    while True:
        yield get_gen_input(shape=shape, minMax = minMax)

def get_gen_out(gen, noiseDim=defaults.NOISE_DIM, batchSize=defaults.BATCH_SIZE, training=False, nTimeSteps=1):
    return gen(
        get_gen_input((batchSize, nTimeSteps, noiseDim)),
        training=training
    )

def get_nBatches_lstm(nBatches, gen, resetStates=True, noiseDim=defaults.NOISE_DIM, batchSize=defaults.BATCH_SIZE, **kwargs):
    genOut = [[] for _ in range(batchSize)]
    for batchNum in range(nBatches):
        oneOut = get_gen_out(gen=gen, noiseDim=noiseDim, batchSize=batchSize, **kwargs)
        oneOut = tf.concat(oneOut, axis=-1)
        for i, arr in enumerate(oneOut):
            genOut[i].append(arr)

    #concat each query into its respective batch. Add new dim to prepare for next concat
    genOut = [
            tf.expand_dims(
                tf.concat(batch, axis=0),
                axis=0
            )
        for batch in genOut]

    #concat batches into one arr
    genOut = tf.concat(genOut, axis=0)

    if resetStates:
        gen.reset_states()
    return genOut

