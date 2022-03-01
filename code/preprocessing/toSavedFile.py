# import pandas as pd
# from utils import meta
# import labels
# import filePaths as fp
#
# _rawPath = fp.rawPath + fp.casasBinHome
# _processedPath = fp.processedPath + fp.casasBinHome
#
#
# def get_debug_data():
#     df = pd.read_csv(_rawPath + "debug.csv")
#     return df
#
# def get_all_homes(path = _processedPath, firstN=None):
#     homePaths = fp.get_all_casas_bin_paths(path)
#     allData = []
#     for hp in homePaths:
#         allData.append(meta.ml_data(
#             train= pd.read_csv(hp.train).iloc[:firstN],
#             test= pd.read_csv(hp.test).iloc[:firstN]
#         ))
#     return allData
#
# def all_homes_concat(firstN = None):
#     homeData = meta.ml_data(pd.DataFrame(columns=labels.features), pd.DataFrame(columns=labels.features))
#     homePaths = fp.get_all_casas_bin_paths(_processedPath)
#     for hp in homePaths:
#         dfTr = pd.read_csv(hp.train)
#         dfTe = pd.read_csv(hp.test)
#         homeData.train = pd.concat((homeData.train, dfTr), axis=0)
#         homeData.test = pd.concat((homeData.test, dfTe), axis=0)
#     homeData.train = homeData.train[:firstN]
#     homeData.test = homeData.test[:firstN]
#     return homeData
#
# def preprocess(df):
#     df[labels.rl.time] = pd.to_datetime(df[labels.rl.time]).values.astype('int64')
#     aOneHot = pd.get_dummies(df[labels.rl.activity])
#     df = pd.concat((df, aOneHot), axis=1)
#     df.drop([labels.rl.activity], axis=1, inplace=True)
#     df = pd.concat((df,pd.get_dummies(df[labels.rl.sensor])), axis=1)
#     df.drop([labels.rl.sensor], axis=1, inplace=True)
#     return df
#
# def preprocess_all():
#     rawHomePaths = fp.get_all_casas_bin_paths(_rawPath)
#     def process(fName, suffix):
#         df = pd.DataFrame(columns=labels.features)
#         dfIr = preprocess(pd.read_csv(fName))
#         dfIr = pd.concat((df, dfIr), axis=0)
#         dfIr.fillna(value=0, inplace=True)
#         dfIr[labels.rl.signal] = dfIr[labels.rl.signal]\
#             .map({labels.motionTrue:1, labels.motionFalse:0, labels.doorTrue:1, labels.doorFalse:0})
#         assert dfIr[labels.rl.signal].isin([0,1]).all()
#         dfIr.to_csv(_processedPath + suffix, index=False)
#         return dfIr
#     for i, hp in enumerate(rawHomePaths):
#         print(i)
#         process(hp.train, "b" + str(i+1) + "Train.csv")
#         process(hp.test, "b" + str(i+1) + "Test.csv")
#     return
#
# def assert_order(allData=None):
#     if allData is None:
#         allData = get_all_homes()
#     for ad in allData:
#         for dataSub in (ad.train, ad.test):
#             assert len(dataSub.columns) == len(labels.features)
#             for c, og in zip(dataSub.columns, labels.features):
#                 assert c == og
#     return
#
#
#
# if __name__ == "__main__":
#     # df = get_all_concat_data()
#     preprocess_all()
#     allData = get_all_homes()
#     assert_order(allData)
#     exit(0)
#
#
# # def proc_df(filePath):
# #     df = pd.read_csv(filePath)
# #     df['Time'] = df['Date'] + ' ' + df['Time']
# #     df.drop(["Date"], axis=1, inplace=True)
# #     df.to_csv(filePath)
# #     return df
#
#
# # def proc(filepath):
# #     df = pd.read_csv(filepath)
# #     df.drop('Unnamed: 0', axis=1, inplace=True)
# #     df.to_csv(filepath, index=False)
# #     return df
