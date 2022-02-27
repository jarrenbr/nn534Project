import copy

from general import meta
import keras
import numpy as np
import pandas as pd

DEBUG = meta.get_debug_flag()

NOISE_DIM = 128
COND_NOISE_DIM=16
BATCH_SIZE = 128
LEAKY_ALPHA_DEFAULT = 0.2
DROPOUT_DEFAULT = 0.2

REG_METRICS = [keras.metrics.RootMeanSquaredError(), keras.losses.MeanAbsolutePercentageError(), keras.losses.CosineSimilarity(),]

TANH = True
NORM_DEFAULT = (-1,1)

SIGMOID = False

#to refactor: TRAIN_IMAGE, TEST_IMAGE, IMG_SHAPE

class np_array_falsifiable(np.ndarray):
    def __init__(self, *args, **kwargs):
        np.ndarray(np_array_falsifiable, self).__new__(*args, **kwargs)

    def __bool__(self):
        return self.length