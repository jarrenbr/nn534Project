from tensorflow import keras

NOISE_DIM = 128
BATCH_SIZE = 128
LEAKY_ALPHA_DEFAULT = 0.2
DROPOUT_DEFAULT = 0.2

MIN_VAL = 0.
MAX_VAL = 1.
MIN_MAX_RNG = (MIN_VAL, MAX_VAL)

METRICS = [keras.metrics.RootMeanSquaredError(), keras.losses.MeanAbsolutePercentageError(), keras.losses.CosineSimilarity(),]

def leaky_relu():
    return keras.layers.LeakyReLU(alpha=LEAKY_ALPHA_DEFAULT)