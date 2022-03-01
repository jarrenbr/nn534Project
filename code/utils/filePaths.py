from dataclasses import dataclass

import common

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

    @staticmethod
    def _file_path(filename):
        return folder.data + "binaryCasas/" + folder.data + filename

    @staticmethod
    def _train_file(prefix):
        return binary_casas._file_path(prefix + binary_casas._trainSuffix)

    @staticmethod
    def _test_file(prefix):
        return binary_casas._file_path(prefix + binary_casas._testSuffix)

    _home_prefix = "b"
    _home1Name = _home_prefix + "1"
    _home2Name = _home_prefix + "2"
    _home3Name = _home_prefix + "3"

    @staticmethod
    def _create_home_data(homeName) -> common.ml_data:
        return common.ml_data(train=binary_casas._train_file(homeName), test=binary_casas._test_file(homeName))

    h1Home = _create_home_data(_home1Name)
    h2Home = _create_home_data(_home2Name)
    h3Home = _create_home_data(_home3Name)

    allHomes = [h1Home, h2Home, h3Home]

@dataclass(frozen=True)
class extensions:
    kerasModel = ".km"
