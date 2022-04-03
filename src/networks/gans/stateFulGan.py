import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from utils import filePaths as fp, globalVars as gv, common
from networks.gans import genApi, wgan
from names import binaryCasasNames as bcNames
from networks import commonBlocks as cBlocks, defaults
from processData.binaryCasasProcess import binaryCasasData as bcData


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
        cBlocks.conv_args(nFilters=bcNames.nGanFeatures, kernelSize=3),
        ]

    x = layers.Conv1DTranspose(**args[0].kwargs)(inputLayer)
    x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True, )
    for arg in args[1:]:
        x = layers.Conv1DTranspose(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True,)

    model = keras.models.Model(inputLayer, x, "LSTM_Generator")
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model


def get_critic() -> keras.models.Model:
    #goal: 32 X 48
    inputLayer = keras.Input(
        shape=(N_TIME_STEPS, bcNames.nGanFeatures)
    )
    args = [
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        cBlocks.conv_args(nFilters=96, kernelSize=3),
        ]
    x = layers.Conv1DTranspose(**args[0].kwargs)(inputLayer)
    x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=False, use_dropout=True, )
    for arg in args[1:]:
        x = layers.Conv1DTranspose(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=False, use_dropout=True)

    model = keras.models.Model(inputLayer, x, "Critic")
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_data():
    return bcData.get_all_homes_as_xy_combined_gen(
        CRITIC_BATCH_SIZE, N_TIME_STEPS, firstN=gv.DATA_AMT)

def run_gan():
    gen = get_generator()
    critic = get_critic()
    data = get_data()
    gan = wgan.wgan(critic, gen, defaults.NOISE_DIM)
    gan.compile()
    # gan.fit(data[0].data.train.gen)
    windows = data[0].data.train.data
    windows = np.reshape(windows[:1024], (-1, N_TIME_STEPS, bcNames.nGanFeatures))
    gan.fit(windows)
    return gan

if __name__ == "__main__":

    if gv.DEBUG:
        common.enable_tf_debug()
    gan = run_gan()

    exit()