from dataclasses import dataclass

import pandas as pd

import common

from names import binaryCasas

@dataclass(frozen=True)
class folder:
    data = "data/"
    raw = "raw/"
    processed = 'processed/'
    misc = "misc/"
    kmModel = misc + "kmModels/"


@dataclass(frozen=True)
class binary_casas:
    _fileExtension = ".csv"
    _trainSuffix = "Train" + _fileExtension
    _testSuffix = "Test" + _fileExtension

    _filePath = folder.data + "binaryCasas/" + folder.data + "{filename}"
    _trainFile = "{filename}" + _trainSuffix
    _testFile = "{filename}" + _testSuffix

    _homeName = "b{}"
    _home1Name = _homeName.format("1")
    _home2Name = _homeName.format("2")
    _home3Name = _homeName.format("3")

    @staticmethod
    def _create_home_data(homeName) -> common.ml_data:
        return common.ml_data(
            train=binary_casas._trainFile.format(filename=homeName),
            test=binary_casas._testFile.format(filename=homeName)
        )



    # h1Home = _create_home_data(_home1Name)
    # h2Home = _create_home_data(_home2Name)
    # h3Home = _create_home_data(_home3Name)
    #
    # allHomes = [h1Home, h2Home, h3Home]

    @staticmethod
    def _get_home(fileNames:common.ml_data):
        return fileNames.transform(pd.read_csv)

    @staticmethod
    def get_home1():
        return binary_casas._get_home(binary_casas.h1Home)

    @staticmethod
    def get_home2():
        return binary_casas._get_home(binary_casas.h2Home)

    @staticmethod
    def get_home3():
        return binary_casas._get_home(binary_casas.h3Home)

    @staticmethod
    def get_all_homes():
        return [binary_casas.get_home1(), binary_casas.get_home2(), binary_casas.get_home3()]

# with binary_casas as bc:
    # bc.h1Home = bc._create_home_data(bc._home1Name)
    # bc.h2Home = bc._create_home_data(bc._home2Name)
    # bc.h3Home = bc._create_home_data(bc._home3Name)
    #
    # allHomes = [bc.h1Home, bc.h2Home, bc.h3Home]


@dataclass(frozen=True)
class extensions:
    kerasModel = ".km"


if __name__ == "__main__":
    exit()