import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

from names import binaryCasasNames as bcNames
from utils import filePaths, common, home
from networks import defaults
from processData.binaryCasasProcess import timeFeat

def _preprocess_time(df:pd.DataFrame):
    df = timeFeat.time_difs(df, bcNames.rl.time)
    df[bcNames.rl.time], trainMaxTime = timeFeat.norm_time(
        df[bcNames.rl.time]
    )
    return df, trainMaxTime


def _preprocess(home:home.home):
    """
    home's data is an ml_data of pd.DataFrame's.
    """

    #time
    home.maxTimeDif = common.ml_data()
    home.data.train, home.maxTimeDif.train = _preprocess_time(home.data.train)
    home.data.test, home.maxTimeDif.test = _preprocess_time(home.data.test)

    return home

def _get_home(getHomeFunc, name, firstN=None):
    h = getHomeFunc()
    h.transform(lambda x: x[:firstN])
    h = home.home(data=h, name = name)
    h = _preprocess(h)
    return h

def get_home1(firstN=None):
    return _get_home(filePaths.binary_casas.get_home1, bcNames.house_names.home1, firstN)

def get_home2(firstN=None):
    return _get_home(filePaths.binary_casas.get_home2, bcNames.house_names.home2, firstN)

def get_home3(firstN=None):
    return _get_home(filePaths.binary_casas.get_home3, bcNames.house_names.home3, firstN)

def get_all_homes(firstN=None):
    return [get_home1(firstN), get_home2(firstN), get_home3(firstN)]

def df_to_window_gen(df:pd.DataFrame, batchSize, nTimeSteps, stride=None, xyPivot=None) -> common.windows_generator:
    assert (df.columns == bcNames.correctOrder).all()
    return common.windows_generator(
        df.to_numpy(), length=nTimeSteps, batchSize=batchSize, stride=stride,
        xyPivot=bcNames.pivots.activities.start
    )



def get_all_homes_as_window_gen(batchSize, nTimeSteps, stride=None, firstN = None, xyPivot=None):
    """
    :param splitXy: do true for classifiers and false for GANs
    """
    homes = get_all_homes(firstN)
    for i in range(len(homes)):
        homes[i].data.train = df_to_window_gen(homes[i].data.train, batchSize, nTimeSteps, stride, xyPivot=xyPivot)
        homes[i].data.test = df_to_window_gen(homes[i].data.test, batchSize, nTimeSteps, stride, xyPivot=xyPivot)

    return homes


if __name__ == "__main__":
    # homes = get_all_homes(100)
    homes = get_all_homes_as_window_gen(8, 4, firstN=100)
    exit()