from tensorflow import keras
from dataclasses import dataclass

from utils import globalVars as gv

#Data shape
NOISE_DIM = 128
BATCH_SIZE = 64

#Data values
MIN_VAL = 0.
MAX_VAL = 1.
MIN_MAX_RNG = (MIN_VAL, MAX_VAL)

#Training
STEPS_PER_EPOCH = 2 if gv.DEBUG else 1000
VALIDATION_STEPS = 2 if gv.DEBUG else 150
PREDICT_STEPS = 40 if gv.DEBUG else 1000


KERAS_FIT_KWARGS = {"steps_per_epoch" : STEPS_PER_EPOCH, "validations_steps" : VALIDATION_STEPS}

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
METRICS = [keras.metrics.RootMeanSquaredError(), keras.losses.CosineSimilarity(),]
