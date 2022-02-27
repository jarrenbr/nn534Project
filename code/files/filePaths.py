from dataclasses import dataclass

#max help me with this pls
@dataclass(frozen=True)
class folder:
    data = "data/"
    raw = data + "raw/"
    processed = data + 'processed/'
    misc = "misc/"
    kmModel = misc + "kmModels/"

@dataclass
class data:
    pass

@dataclass
class file:
    _f = folder
    home1Test = "b1.test.csv"

@dataclass(frozen=True)
class extensions:
    kerasModel = ".km"
