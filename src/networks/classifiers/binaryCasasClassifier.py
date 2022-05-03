#max is so slow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
from pathlib import Path
import pandas as pd
import glob
import numpy as np

from processData.binaryCasasProcess import binaryCasasData as bcData
from names import binaryCasasNames as bcNames
from networks import commonBlocks as cBlocks, defaults
from utils import common, globalVars as gv
from processData import windowsGenerator as wg

BATCH_SIZE = 64
N_TIME_STEPS = 32
EPOCHS = 2 if gv.DEBUG else 6

def run():
    df = getSynthData()
    trainModel(df)

"""
Merge multiple .csv files into one dataframe
"""
def getSynthData():
    synth_dir = Path(__file__).parent.parent.parent/'synthetic-data'
    all_files = glob.glob(str(synth_dir) + "/*.csv")
    print(all_files)
    df_from_each_file = (pd.read_csv(f, header = None) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    return df

def trainModel(df: pd.DataFrame):
    train=df.sample(frac=0.8, random_state=200) #random state is a seed value
    test=df.drop(train.index)
    train_gen = getGen(train)
    test_gen = getGen(test)
    print(df.shape)

def getGen(df: pd.DataFrame) -> wg.windows_generator:
    return bcData.df_to_gen(df, wg.x_y_split_windows, batchSize = BATCH_SIZE, nTimesteps = N_TIME_STEPS, xyPivot = bcNames.pivots.activities.start, stride = None)



def basic_cnn() -> keras.models.Model:
    inputLayer = keras.Input(shape=(N_TIME_STEPS, len(bcNames.features))) #should be 16 X 48
    x = cBlocks.conv_block(inputLayer, 20, defaults.leaky_relu(),) #8 timesteps
    x = cBlocks.conv_block(x, 12, defaults.leaky_relu()) #4 timesteps
    x = cBlocks.conv_block(x, 8, defaults.leaky_relu()) #2 timesteps
    x = cBlocks.conv_block(x, len(bcNames.allActivities), activation=keras.activations.softmax)
    x = l.Flatten()(x)
    model = keras.models.Model(inputLayer, x, name="Basic_CNN_Classifier")
    model.compile(loss = keras.losses.CategoricalCrossentropy(),
                  optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

#Nathan uses temporal CNNs

def multi_head_cnn():
    pass

def run_classifiers(data:common.ml_data):
    model = basic_cnn()
    # print(model.summary())
    history = model.fit(data.train.gen, epochs=EPOCHS, steps_per_epoch = defaults.STEPS_PER_EPOCH,
                        validation_data=data.test.gen, validation_steps=defaults.VALIDATION_STEPS)
    preds = model.predict(data.test.gen, steps=defaults.PREDICT_STEPS)
    predsOneHot = np.zeros(preds.shape)
    predsOneHot[
        preds.argmax(axis=1).reshape((-1,1))
    ] = 1
    return model, history

if __name__ == "__main__":
    allHomes = bcData.get_all_homes_as_xy_split_gen(
        batchSize=BATCH_SIZE, nTimesteps=N_TIME_STEPS,
        xyPivot=bcNames.pivots.activities.start, firstN=gv.DATA_AMT
    )
    run_classifiers(allHomes[0].data)
    # run()
    exit()