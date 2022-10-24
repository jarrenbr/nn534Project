import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosSimF
import tensorflow as tf

from names import binaryCasasNames as bcNames

def to_one_hot(arr):
    return tf.one_hot(
        tf.math.argmax(arr, axis=-1),
        arr.shape[-1]).numpy()

def enforce_alt_binary(arr):
    option1 = np.tile((0., 1.), math.ceil(arr.shape[0]/2))
    option2 = np.tile((1., 0.), math.ceil(arr.shape[0]/2))
    if arr.shape[0] % 2:
        option1, option2 = option1[:-1], option2[:-1]

    cosSim1 = cosSimF([arr], [option1])
    # cosSim2 = cosSimF([arr], [option2])
    assert cosSim1.shape == (1,1)
    return option1 if cosSim1 > 0 else option2


def enforce_alt_signal_each_sensor(arr3d:np.ndarray,)-> np.ndarray:
    assert len(arr3d.shape) == 3
    #sensors must be one-hot encoded
    for batchNum in range(arr3d.shape[0]):
        for sensNum in range(bcNames.pivots.sensors.start, bcNames.pivots.sensors.stop):
            mask = arr3d[batchNum, :, sensNum] == 1
            if mask.any():
                arr3d[batchNum, mask, bcNames.pivots.signal.start] = \
                    enforce_alt_binary(arr3d[batchNum, mask, bcNames.pivots.signal.start])
    return arr3d


def gen_out_one_hot_sensor(genOut:np.ndarray)->np.ndarray:
    #one hot sensor
    genOut[..., bcNames.pivots.sensors.start:bcNames.pivots.sensors.stop] =\
        to_one_hot(genOut[..., bcNames.pivots.sensors.start:bcNames.pivots.sensors.stop])
    return genOut

def gen_out_to_real_normalized(genOut:np.ndarray) -> np.ndarray:
    #one hot sensor
    genOut = gen_out_one_hot_sensor(genOut)
    #one hot activity
    genOut[..., bcNames.pivots.activities.start:] = to_one_hot(genOut[..., bcNames.pivots.activities.start:])

    #binarize signal
    genOut = enforce_alt_signal_each_sensor(genOut)
    return genOut


