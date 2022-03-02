from tensorflow import keras
from dataclasses import dataclass

#Data shape
NOISE_DIM = 128
BATCH_SIZE = 128

#Data values
MIN_VAL = 0.
MAX_VAL = 1.
MIN_MAX_RNG = (MIN_VAL, MAX_VAL)


#Hyper-parameters
DROPOUT_PORTION = 0.2

@dataclass(frozen=True)
class adam:
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.9
    kwargs = {"learning_rate" : lr, "beta_1" : beta1, "beta_2" : beta2}

def optimizer(kwargs = adam.kwargs):
    return keras.optimizers.Adam(**kwargs)

LEAKY_ALPHA = 0.2
def leaky_relu():
    return keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

#Other
METRICS = [keras.metrics.RootMeanSquaredError(), keras.losses.MeanAbsolutePercentageError(), keras.losses.CosineSimilarity(),]
