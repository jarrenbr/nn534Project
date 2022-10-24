import pandas as pd
from sklearn import preprocessing as skpp

maxTimeDif = pd.Timedelta(days=7)

def time_difs(df:pd.DataFrame, timeCol):
    df[timeCol] = df[timeCol].diff()
    df.drop(0, axis=0, inplace=True)
    return df

# _timeExp = .3
_timeExp = .15

def norm_time(arr, scaler=None):
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    a = arr.reshape(-1,1)
    if scaler is None:
        scaler = skpp.QuantileTransformer()
        a = scaler.fit_transform(a)
    else:
        a = scaler.transform(a)
    assert not (a < 0).any()
    assert not (a > 1).any()
    return a, scaler



def unnorm_time(arr, scaler):
    arr = scaler.inverse_transform(arr)
    assert (arr >= 0).all()
    return arr


#todo: add time since midnight and weekday?