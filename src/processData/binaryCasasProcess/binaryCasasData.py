import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

from names import binaryCasasNames as bcNames
from utils import filePaths, common, home, globalVars as gv
from networks import defaults
from processData.binaryCasasProcess import timeFeat
from processData import windowsGenerator as wg

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
    h = home.home(data=h, name = name)
    h = _preprocess(h)
    h.data.transform(lambda x: x[:firstN])
    return h

def get_home1(firstN=None):
    return _get_home(filePaths.binary_casas.get_home1, bcNames.house_names.home1, firstN)

def get_home2(firstN=None):
    return _get_home(filePaths.binary_casas.get_home2, bcNames.house_names.home2, firstN)

def get_home3(firstN=None):
    return _get_home(filePaths.binary_casas.get_home3, bcNames.house_names.home3, firstN)

def get_all_homes(firstN=None):
    return [get_home1(firstN), get_home2(firstN), get_home3(firstN)]

def df_to_gen(df:pd.DataFrame, windowClass, **kwargs):
    assert (df.columns == bcNames.correctOrder).all()
    return windowClass(data=df.to_numpy(), **kwargs)

def _get_all_homes_as_gen(firstN, windowCls, **kwargs):
    homes = get_all_homes(firstN)
    for i in range(len(homes)):
        homes[i].data.train = df_to_gen(homes[i].data.train, windowCls, **kwargs,)
        homes[i].data.test = df_to_gen(homes[i].data.test, windowCls, **kwargs, )
    return homes

def get_all_homes_as_xy_combined_gen(batchSize, nTimesteps, stride=None, firstN = gv.DATA_AMT):
    return _get_all_homes_as_gen(firstN, wg.x_y_concat_windows, batchSize=batchSize, nTimesteps=nTimesteps, stride=stride)

def get_all_homes_as_xy_split_gen(batchSize, nTimesteps, stride=None, xyPivot=bcNames.pivots.activities.start, firstN = gv.DATA_AMT):
    return _get_all_homes_as_gen(firstN, wg.x_y_split_windows, xyPivot=xyPivot,
                                 batchSize=batchSize, nTimesteps=nTimesteps, stride=stride)

if __name__ == "__main__":
    # homes = get_all_homes(100)
    homes = get_all_homes_as_xy_combined_gen(8, 4, firstN=100)
    homes = get_all_homes_as_xy_split_gen(8,4,firstN=100)
    exit()