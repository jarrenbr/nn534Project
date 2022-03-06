import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import files.filePaths as fp
from networks.gans import genApi, wgan
from names import binaryCasasNames as bcNames
from networks import commonBlocks as cBlocks, defaults

MODEL_DIR = fp.folder.kmModel + "statefulGan/"
# def classifier_naming

GENERATOR_FILE = MODEL_DIR + "generator" + fp.extensions.kerasModel
CRITIC_FILE = MODEL_DIR + "critic" + fp.extensions.kerasModel

GENERATOR_BATCH_SIZE = 16
CRITIC_BATCH_SIZE = 128
N_TIME_STEPS = 32
NOISE_DIM = 128

def get_generator() -> keras.models.Model:
    #goal: 32 X 48
    inputLayer = keras.Input(
        shape=(1, NOISE_DIM)
    )
    args = [
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        ]
    for arg in args:
        x = layers.Conv1DTranspose(
            filters=arg.nFilters, kernel_size=arg.kernelSize, use_bias=False
        )(inputLayer)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True,)

    model = keras.models.Model(inputLayer, x, "LSTM_Generator")


if __name__ == "__main__":
    gan = wgan.wgan()
    exit()