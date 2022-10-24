import tensorflow as tf
from tensorflow import keras


class SliceInnerMost(keras.layers.Layer):
    def __init__(self, begin, end,**kwargs):
        super(SliceInnerMost, self).__init__(**kwargs)
        self.begin = begin
        self.end = end

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'end': self.end,
        })
        return config
    def call(self, inputs):
        return inputs[...,self.begin: self.end]
