import copy

miscDir = "misc/"
dataDir = "datasets/"

import sys
def get_debug_flag():
    return sys.gettrace() is not None

DEBUG = get_debug_flag()

def tf_gpu_cap(percent=.9):
    import tensorflow as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=percent)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class keras_params:
    def __init__(self, runEpochs):
        debug = get_debug_flag()
        self.verbose = 1 if debug else 0
        self.epochs = 1 if debug else runEpochs
        self.kpKwArgs = {"verbose": self.verbose, "epochs": self.epochs}

def enable_tf_debug(eager: object = True, debugMode: object = True) -> object:
    import tensorflow as tf
    tf.config.run_functions_eagerly(eager)
    if debugMode: tf.data.experimental.enable_debug_mode()

def tf_np_behavior():
    import tensorflow.python.ops.numpy_ops.np_config as np_config
    np_config.enable_numpy_behavior()

class x_y:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

def split_to_xy(data, yStart):
    if isinstance(data, list):
        xys = type(data)()
        for d in data:
            xys.append(x_y(x=d[...,:yStart], y=d[...,yStart:]))
        return xys
    return x_y(x=data[...,:yStart], y=data[...,yStart:])


class ml_data:
    def __init__(self, train=None, test=None, validate=None):
        self.train = train
        self.test = test
        self.validate = validate


    def apply(self, function, *args, **kwargs):
        returns = ml_data()
        if self.train is not None:
            returns.train = function(self.train, *args, **kwargs)
        if self.test is not None:
            returns.test = function(self.test, *args, **kwargs)
        if self.validate is not None:
            returns.validate = function(self.validate, *args, **kwargs)
        return returns

    def transform(self, function, *args, **kwargs):
        if self.train is not None:
            self.train = function(self.train, *args, **kwargs)
        if self.test is not None:
            self.test = function(self.test, *args, **kwargs)
        if self.validate is not None:
            self.validate = function(self.validate, *args, **kwargs)

class time_shape_base:
    def __init__(self, nTimeSteps=None, nLabels=None, nFeatures=None, nGanFeatures=None, nSamples=None):
        self.nTimeSteps = nTimeSteps
        self.nGanFeatures = nGanFeatures
        self.nLabels = nLabels
        self.nFeatures = nFeatures
        self.nSamples = nSamples

class time_series_shape(time_shape_base):
    def __init__(self, x:tuple, y:tuple=None, makeYCompliant:bool=None, nSamples = None,
                 nTimeSteps=None, nFeatures=None, nLabels=None
         ):
        # assert len(x) == 3
        if nSamples is None: nSamples = x[0]
        if nTimeSteps is None: nTimeSteps = x[1]
        if nFeatures is None: nFeatures = x[2]
        self.x = (nSamples, nTimeSteps, nFeatures)
        nGanFeatures = None, None
        if y is not None or nLabels is not None:
            if nLabels is None: nLabels = y[-1]
            nGanFeatures = nLabels + nFeatures
            self.y = (self.x[0], self.x[1], self.nLabels) if makeYCompliant or y is None else y
        super.__init__(nTimeSteps, nLabels, nFeatures, nGanFeatures, nSamples)


def divide_min(dividend, divisor, minimum, castType=int):
    return castType(max(dividend/divisor), minimum)


#if cuda is working with tensorflow, this sets gpu0 to NOT be visible
def disable_gpu():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

#if cuda is working with tensorflow, this sets gpu0 to be visible
def enable_gpu():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"