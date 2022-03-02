#max is so slow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l

from processData.binaryCasasProcess import binaryCasasData as bcData
from names import binaryCasasNames as bcNames
from networks import commonBlocks as cBlocks, defaults
from utils import common, globalVars as gv

BATCH_SIZE = 64
N_TIME_STEPS = 32

def basic_cnn() -> keras.models.Model:
    inputLayer = keras.Input(shape=(N_TIME_STEPS, len(bcNames.features))) #should be 48 channels
    x = cBlocks.conv_block(inputLayer, 24, defaults.leaky_relu(),) #16 timesteps
    x = cBlocks.conv_block(x, 15, defaults.leaky_relu()) #8 timesteps
    x = cBlocks.conv_block(x, 11, defaults.leaky_relu()) #4 timesteps
    x = cBlocks.conv_block(x, 8, defaults.leaky_relu()) #2 timesteps
    x = cBlocks.conv_block(x, len(bcNames.allActivities), activation=keras.activations.softmax)
    x = l.Flatten()(x)
    model = keras.models.Model(inputLayer, x, name="Basic_CNN_Classifier")
    return model


def multi_head_cnn():
    pass

def run_classifiers(data:common.ml_data):
    model = basic_cnn()
    model.compile(loss = keras.losses.CategoricalCrossentropy(), optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    # print(model.summary())
    history = model.fit(data.train.gen, epochs=5, steps_per_epoch = defaults.STEPS_PER_EPOCH,
                        validation_data=data.test.gen, validation_steps=defaults.VALIDATION_STEPS)
    return model, history

if __name__ == "__main__":
    allHomes = bcData.get_all_homes_as_window_gen(
        batchSize=BATCH_SIZE, nTimeSteps=N_TIME_STEPS, xyPivot=bcNames.pivots.activities.start, firstN=gv.DATA_AMT
    )
    run_classifiers(allHomes[0].data)


    exit()