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

GENERATOR_TIME_STEPS = 16
CRITIC_TIME_STEPS = 128
NOISE_DIM = 128
BATCH_SIZE = defaults.BATCH_SIZE

def get_conv_generator() -> keras.models.Model:
    #goal: 16 X 48
    inputLayer = keras.Input(
        shape=(1,NOISE_DIM,)
    )

    args = [
        cBlocks.conv_args(nFilters=100, kernelSize=4),
        cBlocks.conv_args(nFilters=75, kernelSize=2),
        cBlocks.conv_args(nFilters=bcNames.nGanFeatures, kernelSize=2),
        ]

    x = layers.Conv1DTranspose(**args[0].kwargs)(inputLayer)
    x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True, )
    for arg in args[1:]:
        x = layers.Conv1DTranspose(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True,)
    x = layers.Dense(bcNames.nGanFeatures, keras.activations.tanh)(x)

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= "Conv_Generator")
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_lstm_generator(batchSize=defaults.BATCH_SIZE) -> keras.models.Model:
    # goal: 16 X 48
    inputLayer = keras.Input(
        batch_shape=(batchSize, 1, NOISE_DIM,)
    )

    nFilters = 100

    args = [
        cBlocks.conv_args(nFilters=nFilters, kernelSize=4),
        cBlocks.conv_args(nFilters=nFilters, kernelSize=2),
        cBlocks.conv_args(nFilters=nFilters, kernelSize=2),
    ]

    x = layers.Conv1DTranspose(**args[0].kwargs)(inputLayer)
    x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True, )

    for arg in args[1:]:
        x = layers.Conv1DTranspose(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True, )

    x = layers.Bidirectional(
        layers.LSTM(
            bcNames.nGanFeatures,
            dropout=defaults.DROPOUT_PORTION,
            stateful=True,
            return_sequences=True
        )
    )(x)
    x = layers.Dense(bcNames.nGanFeatures, keras.activations.tanh)(x)

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name="LSTM_Generator")
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_critic() -> keras.models.Model:
    #128 X 48
    inputLayer = keras.Input(
        shape=(CRITIC_TIME_STEPS, bcNames.nGanFeatures)
    )
    x = layers.Dense(bcNames.nGanFeatures, defaults.leaky_relu())(inputLayer)

    args = [
        cBlocks.conv_args(nFilters=100, kernelSize=8, strides=4),
        cBlocks.conv_args(nFilters=150, kernelSize=4),
        cBlocks.conv_args(nFilters=150, kernelSize=4),
        cBlocks.conv_args(nFilters=160, kernelSize=4),
        ]
    x = layers.Conv1D(**args[0].kwargs)(x)
    x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=False, use_dropout=True, )
    for arg in args[1:]:
        x = layers.Conv1D(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=False, use_dropout=True)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= "Conv_Critic")
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_data(batchSize = BATCH_SIZE):
    return bcData.get_all_homes_as_xy_combined_gen(
        batchSize, CRITIC_TIME_STEPS, firstN=gv.DATA_AMT)

def run_gan():
    gen = get_lstm_generator(batchSize=32)
    critic = get_critic()
    data = get_data()
    gan = wgan.wgan(
        critic, gen, defaults.NOISE_DIM, nCriticTimesteps=CRITIC_TIME_STEPS, nGenTimesteps=GENERATOR_TIME_STEPS
    )
    gan.compile()
    # gan.fit(data[0].data.train.gen)
    windows = data[0].data.train.data
    validSize = windows.shape[0] - (windows.shape[0] % (CRITIC_TIME_STEPS * bcNames.nGanFeatures))
    windows = np.reshape(windows[:validSize], (-1, CRITIC_TIME_STEPS, bcNames.nGanFeatures))
    gan.fit(windows)
    return gan

if __name__ == "__main__":

    if gv.DEBUG:
        common.enable_tf_debug()
    gan = run_gan()

    exit()