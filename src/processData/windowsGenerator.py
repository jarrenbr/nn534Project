import numpy as np
import tensorflow as tf

class windows_generator():
    """
    Call next(self.gen) to slide the window.
    """
    def __init__(self, data:np.ndarray, batchSize, nTimesteps, stride=None):
        self.data = data
        self.stride = nTimesteps if stride is None else stride
        self.batchSize = batchSize
        self.nTimesteps = nTimesteps
        nBatches = int(data.shape[0] / nTimesteps)
        self.nSamplesPerBatch = int(nBatches / batchSize)

        self.gen = self._gen_init()
        self.init_index()

    def init_index(self):
        initPositions = np.linspace(0, self.data.shape[0], num=self.batchSize, dtype=int, endpoint=False).reshape((-1,1))
        initPositions = np.repeat(initPositions, self.nTimesteps, axis=1)
        rng = np.arange(initPositions.shape[-1]).reshape((-1,1))
        self.currIndex = np.repeat(rng, initPositions.shape[0], axis=1).T
        self.currIndex += initPositions
        return

    def reset_index(self, rndShift=True):
        self.init_index()
        if rndShift:
            shift = np.random.randint(0, self.nTimesteps)
            self.currIndex += shift

    def xy_split(self):
        x, y = np.split(
            self.data[self.currIndex],
            [self.xyPivot],
            axis=-1
        )
        return x, y[:,-1]

    def no_split(self):
        return self.data[self.currIndex]

    def get_data(self):
        return tf.convert_to_tensor(self.data[self.currIndex])

    def _gen_init(self):
        while True:
            yield self.get_data()
            self.currIndex += self.stride
            if self.currIndex[-1, -1] >= self.data.shape[0]:
                self.reset_index()

class x_y_split_windows(windows_generator):
    def __init__(self, xyPivot, **kwargs):
        super(x_y_split_windows, self).__init__(**kwargs)
        self.xyPivot = xyPivot

    def get_data(self):
        x, y = np.split(
            self.data[self.currIndex],
            [self.xyPivot],
            axis=-1
        )
        return x, y[:,-1]

class x_y_concat_windows(windows_generator):
    pass


# import math

# class windows_generator(tf.keras.utils.Sequence):
#     pass
#
# class x_y_split_windows(windows_generator):
#     pass
#
# class x_y_concat_windows(windows_generator):
#     def __init__(self, data, batchSize, nTimesteps, offset=None, *args, **kwargs):
#         self.data = data
#         self.nTimesteps = nTimesteps
#         self.batchSize = batchSize
#         self.offset=offset
#
#     def __len__(self):
#         return math.ceil(len(self.data) / (self.batchSize*self.nTimesteps))
#
#     def __getitem__(self, idx):
#         steps = self.batchSize * self.nTimesteps
#         return self.data[idx * steps:(idx+1)*steps].reshape((self.batchSize, self.nTimesteps, -1))
