from utils import globalVars as gv
import numpy as np

def tf_gpu_cap(percent=.9):
    import tensorflow as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=percent)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class keras_params:
    def __init__(self, runEpochs):
        debug = gv.get_debug_flag()
        self.verbose = 1 if debug else 0
        self.epochs = 1 if debug else runEpochs
        self.kpKwArgs = {"verbose": self.verbose, "epochs": self.epochs}

def enable_tf_debug(eager = True, debugMode = True):
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

class windows_generator:
    """
    Call next(self.gen) to slide the window.
    """
    def __init__(self, data:np.ndarray, batchSize, length, stride=None, xyPivot=None, splitXy=False):
        self.data = data
        self.stride = length if stride is None else stride
        self.batchSize = batchSize
        self.length = length

        self.xyPivot=xyPivot
        self.splitXy = splitXy

        self.reset_generator()

    def reset_generator(self):
        initPositions = np.linspace(0, self.data.shape[0], num=self.batchSize, dtype=int, endpoint=False).reshape((-1,1))
        initPositions = np.repeat(initPositions, self.length, axis=1)
        rng = np.arange(initPositions.shape[-1]).reshape((-1,1))
        self.currIndex = np.repeat(rng, initPositions.shape[0], axis=1).T
        self.currIndex += initPositions
        self.gen = self._gen_init()
        return

    def _gen_init(self):
        #for classifiers
        if self.splitXy:
            assert self.xyPivot is not None
            while self.currIndex[-1,-1] < self.data.shape[0]:
                x, y = np.split(
                    self.data[self.currIndex],
                    [self.xyPivot],
                    axis=-1
                )
                yield x, y[:,-1] #choose final activity as label
                self.currIndex += self.stride
        #for gans
        else:
            while self.currIndex[-1, -1] < self.data.shape[0]:
                yield self.data[self.currIndex]
                self.currIndex += self.stride

        self.reset_generator()

