import pandas as pd
import tensorflow as tf
import numpy as np

from names import binaryCasasNames as bcNames
from utils import filePaths, common, home, globalVars as gv
from utils.common import ml_data
from networks import defaults
from processData.binaryCasasProcess import timeFeat
from processData import windowsGenerator as wg

def _preprocess_time(df:pd.DataFrame, scaler=None):
    timeName = bcNames.rl.time
    df[timeName] = pd.to_datetime(df[timeName])
    t = df[timeName]
    doy = "doy"
    secondsOfDay = "secsFrom12Am"

    df[doy] = t.dt.dayofyear / (365 + t.dt.is_leap_year)

    df[secondsOfDay] = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
    df[secondsOfDay] /= pd.Timedelta(days=1).total_seconds()

    toTrig = [doy, secondsOfDay]
    df[toTrig] = df[toTrig] * 2 * np.pi

    pt = bcNames.ProcessedTime
    cosined = np.cos(df[toTrig])
    cosined.rename(columns={doy: pt.doyCos, secondsOfDay: pt.timeDayCos}, inplace=True)
    sined = np.sin(df[toTrig])
    sined.rename(columns={doy: pt.doySin, secondsOfDay: pt.timeDaySin}, inplace=True)

    sined = (sined + 1) / 2
    cosined = (cosined + 1) / 2

    df.drop(columns=toTrig, inplace=True)
    df = pd.concat((df, cosined, sined), axis=1)

    df[pt.timeDif] = df[timeName].diff()
    df.drop(0, axis=0, inplace=True)
    # df[pt.timeDif], scaler = timeFeat.norm_time(
    #     df[pt.timeDif], scaler
    # )
    df.drop(columns=[bcNames.rl.time], inplace=True)
    df = df[bcNames.correctOrder]
    return df


def _preprocess(home:home.home):
    """
    home's data is an ml_data of pd.DataFrame's.
    """
    #time
    home.timeDifScaler = common.ml_data()
    home.data.train= _preprocess_time(home.data.train)
    home.data.test = _preprocess_time(home.data.test)

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

def df_to_arr(df:pd.DataFrame, nTimesteps:int, nFeatures, scaler):
    tDif = bcNames.ProcessedTime.timeDif
    df[tDif] = scaler.transform(df[tDif].to_numpy().reshape(-1,1))
    nTrain = df.shape[0]
    nTrain = nTrain - (nTrain % nTimesteps)
    data = df.iloc[:nTrain].to_numpy().reshape((-1, nTimesteps, nFeatures))
    return data

def get_all_homes_xy_combined(nTimesteps, aggregated=False, firstN=gv.DATA_AMT)->list[home.home]:
    homes = get_all_homes(firstN)
    timeDifs = []
    for i in range(len(homes)):
        timeDifs.append(homes[i].data.train[bcNames.ProcessedTime.timeDif])
        timeDifs.append(homes[i].data.test[bcNames.ProcessedTime.timeDif])
    timeDifs = np.concatenate(timeDifs, axis=0)
    timeDifs, scaler = timeFeat.norm_time(timeDifs)

    for i in range(len(homes)):
        homes[i].data.transform(df_to_arr, nTimesteps, bcNames.nGanFeatures, scaler)
        homes[i].timeDifScaler = scaler

    if aggregated:
        homes = [home.home(
            data = ml_data(
                train= np.concatenate([homes[i].data.train for i in range(len(homes))]),
                test= np.concatenate([homes[i].data.test for i in range(len(homes))]),
            ),
            name = "AggregatedHome",
            timeDifScaler = scaler,
        )]


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