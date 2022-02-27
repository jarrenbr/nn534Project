from tensorflow import keras

NOISE_DIM = 128
COND_NOISE_DIM=16
BATCH_SIZE = 128
LEAKY_ALPHA_DEFAULT = 0.2
DROPOUT_DEFAULT = 0.2

REG_METRICS = [keras.metrics.RootMeanSquaredError(), keras.losses.MeanAbsolutePercentageError(), keras.losses.CosineSimilarity(),]

