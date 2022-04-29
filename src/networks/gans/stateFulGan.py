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


MODEL_DIR = fp.folder.kmModel + "wgan/"
# def classifier_naming

LSTM_GENERATOR_NAME = "LSTM_Generator"
CNN_GENERATOR_NAME = "CNN_Generator"
CRITIC_NAME = "CNN_Critic"

LSTM_GENERATOR_FILE = MODEL_DIR + LSTM_GENERATOR_NAME
CNN_GENERATOR_FILE = MODEL_DIR
CRITIC_FILE = MODEL_DIR + CRITIC_NAME

GENERATOR_TIME_STEPS = 16
CRITIC_TIME_STEPS = 128
NOISE_DIM = 128
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1000

NPREV_EPOCHS_DONE = 0
# NPREV_EPOCHS_DONE = 12
NEPOCHS = 2 if gv.DEBUG else 2
# NEPOCHS = 0

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

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= CNN_GENERATOR_NAME)
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_lstm_generator(batchSize=BATCH_SIZE) -> keras.models.Model:
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

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name=LSTM_GENERATOR_NAME)
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_critic() -> keras.models.Model:
    #128 X 48
    inputLayer = keras.Input(
        shape=(CRITIC_TIME_STEPS, bcNames.nGanFeatures)
    )
    x = layers.GaussianNoise(.025)(inputLayer)
    x = layers.Dense(bcNames.nGanFeatures, defaults.leaky_relu())(x)

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

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= CRITIC_NAME)
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def get_data(batchSize = BATCH_SIZE):
    return bcData.get_all_homes_as_xy_combined_gen(
        batchSize, CRITIC_TIME_STEPS, firstN=gv.DATA_AMT)

def train_on_house(gan, house):
    windows = house.data.train.data

    nOmitted = (windows.shape[0] % (BATCH_SIZE * CRITIC_TIME_STEPS))
    validSize = windows.shape[0] - nOmitted
    # print("House {}. Used:Omitted = {}:{}".format(house.name, validSize, nOmitted))
    assert validSize > 0

    # random offset if some are omitted
    # randomly choose to omit head or tail
    if nOmitted:
        offset = np.random.randint(0, min(CRITIC_TIME_STEPS, nOmitted))
        windows = windows[nOmitted-offset:-offset] if np.random.randint(0,2) else windows[offset:validSize+offset]
    else:
        windows = windows[nOmitted:] if np.random.randint(0,2) else windows[:validSize]


    windows = np.reshape(
        windows,
        (-1, CRITIC_TIME_STEPS, bcNames.nGanFeatures)
    )
    gan.fit(windows, batch_size=BATCH_SIZE, shuffle=False,)
    gan.reset_states()

    return gan

def get_gan(loadGan=False):
    if loadGan:
        gen = keras.models.load_model(LSTM_GENERATOR_FILE)
        critic = keras.models.load_model(CRITIC_FILE)
    else:
        gen = get_lstm_generator()
        critic = get_critic()

    gan = wgan.wgan(
        critic, gen, defaults.NOISE_DIM, nCriticTimesteps=CRITIC_TIME_STEPS, nGenTimesteps=GENERATOR_TIME_STEPS,
        batchSize=BATCH_SIZE,
    )
    gan.compile()
    return gan

def run_gan(gan):
    data = get_data()

    for epoch in range(NEPOCHS):
        printMsg ="Epoch {}/{}".format(epoch, NEPOCHS)
        if NPREV_EPOCHS_DONE:
            printMsg += " ({} done prior)".format(NPREV_EPOCHS_DONE)
        print(printMsg)

        for house in data[::-1]:
            gan = train_on_house(gan, house)

    if not gv.DEBUG:
        gan.save(genFilePath=LSTM_GENERATOR_FILE, criticFilePath=CRITIC_FILE)

    gan.plot_losses_mult_samples_epoch(
        3,
        None if gv.DEBUG else fp.folder.statefulGanImg + "W0Losses",#_CriticNoise",
        NPREV_EPOCHS_DONE
    )

    return gan

if __name__ == "__main__":

    if gv.DEBUG:
        common.enable_tf_debug()

    # loadGan = True
    loadGan = False
    gan = get_gan(loadGan)
    # gan = run_gan(gan)

    genOut = []
    for sampleNum in range(10):
        genOut.append(genApi.get_gen_out(gan.generator, NOISE_DIM, batchSize=BATCH_SIZE))

    #np.ndarray in shape (samples, time steps, features)
    genOut = np.concatenate(genOut, axis=0)


    x,y = genOut[...,:bcNames.nFeatures], genOut[...,-1,bcNames.nFeatures:]

    print(x.shape, y.shape)

    if gv.DEBUG:
        plt.show()
    exit()