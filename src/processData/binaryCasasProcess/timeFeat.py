import pandas as pd


def time_difs(df:pd.DataFrame, timeCol):
    df[timeCol] = df[timeCol].diff()
    df.drop(0, axis=0, inplace=True)
    return df

_timeExp = .3
# _timeExp = .09

def norm_time(arr):
    arr -= arr.min()
    arrMax = arr.max()
    arr /= arrMax
    arr = 2*arr**_timeExp - 1
    assert not (arr < -1).any()
    assert not (arr > 1).any()
    return arr, arrMax



def unnorm_time(arr, oldMax):
    arr = ((arr+1)/2)**(1/_timeExp) * oldMax
    assert (arr >= 0).all()
    return arr


#todo: add time since midnight and weekday?