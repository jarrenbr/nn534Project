import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from utils import filePaths as fp, globalVars as gv, common
from utils.common import ml_data
from networks.gans import genApi, wgan
from names import binaryCasasNames as bcNames
from networks import commonBlocks as cBlocks, defaults
from processData.binaryCasasProcess import binaryCasasData as bcData, postProcess as postProc

MODEL_DIR = fp.folder.kmModel + "wgan/"
IMG_FOLDER = fp.folder.img + "statefulGan/"
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
STEPS_PER_EPOCH = 2 if gv.DEBUG else 250

NEPOCHS = 2 if gv.DEBUG else 2
NPREV_EPOCHS_DONE = 0
# NPREV_EPOCHS_DONE = NEPOCHS

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

#can't save output in graph mode during train step. issue for passing context
#for assigning states, the gradient tape was messed up.
def get_lstm_critic(batchSize=BATCH_SIZE) -> tuple:
    def kernel_regularizer():
        return keras.regularizers.L2(0.01)
    #128 X 48
    inputLayer = keras.Input(
        batch_shape=(batchSize, CRITIC_TIME_STEPS, bcNames.nGanFeatures)
    )
    x = layers.Bidirectional(layers.LSTM(
            bcNames.nGanFeatures,
            return_sequences=False,
            name="criticLstm",
        )
    )(inputLayer)

    x = layers.BatchNormalization()(x)


    nDenseUnits = 250

    x = layers.Dense(nDenseUnits,)(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(nDenseUnits,)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(nDenseUnits,)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1,)(x)

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= CRITIC_NAME)
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    return model

def nn_diagram(model, imgFile=None):
    if imgFile is None:
        imgFile = IMG_FOLDER + model.name + ".png"
    keras.utils.plot_model(model, to_file=imgFile,
                              show_shapes=True,
                              show_layer_activations=True,
                              show_dtype=True,
                              show_layer_names=True)

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
            stateful=True,
            return_sequences=True
        )
    )(x)
    x = layers.BatchNormalization()(x)
    time = layers.Conv1D(len(bcNames.ProcessedTime.order), kernel_size=4, padding='causal',
                         activation=keras.activations.hard_sigmoid)(x)
    # time = layers.Dense(len(bcNames.ProcessedTime.order), keras.activations.hard_sigmoid)(x)
    sensors = layers.Conv1D(len(bcNames.allSensors), kernel_size=3, padding='causal',
                         activation=keras.activations.softmax)(x)
    # sensors = layers.Dense(len(bcNames.allSensors), keras.activations.softmax)(x)


    x = layers.Concatenate()([time, sensors])

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name=LSTM_GENERATOR_NAME)
    return model

def get_critic() -> keras.models.Model:
    #128 X 48
    # timeInput = keras.Input(
    #     shape=(CRITIC_TIME_STEPS, len(bcNames.ProcessedTime.order))
    # )
    # sensorInput = keras.Input(
    #     shape=(CRITIC_TIME_STEPS, len(bcNames.allSensors))
    # )

    inputLayer = keras.Input(
        shape=(CRITIC_TIME_STEPS, bcNames.nGanFeatures)
    )
    x = layers.GaussianNoise(.025)(inputLayer)
    x = layers.Dense(bcNames.nGanFeatures, defaults.leaky_relu())(x)

    padding='causal'
    args = [
        cBlocks.conv_args(nFilters=100, kernelSize=8, strides=4, padding=padding),
        cBlocks.conv_args(nFilters=150, kernelSize=4, padding=padding),
        cBlocks.conv_args(nFilters=150, kernelSize=4, padding=padding),
        cBlocks.conv_args(nFilters=160, kernelSize=4, padding=padding),
        ]
    # x = layers.Conv1D(**args[0].kwargs)(x)
    # x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=False, use_dropout=True, )
    for arg in args:
        x = layers.Conv1D(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=False, use_dropout=True)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= CRITIC_NAME)
    return model

def get_data_gen(batchSize = BATCH_SIZE):
    return bcData.get_all_homes_as_xy_combined_gen(
        batchSize, CRITIC_TIME_STEPS, firstN=gv.DATA_AMT)

def get_data():
    data = bcData.get_all_homes_xy_combined(nTimesteps=CRITIC_TIME_STEPS, aggregated=True)
    return data


def get_gan(loadGan=False):
    if loadGan:
        gen = keras.models.load_model(LSTM_GENERATOR_FILE)
        critic = keras.models.load_model(CRITIC_FILE)
    else:
        gen = get_lstm_generator()
        critic = get_critic()

    gan = wgan.wgan(
        critic, gen, defaults.NOISE_DIM, batchSize=BATCH_SIZE,
    )
    gan.compile()
    return gan

def train_on_house_individually(gan, house):
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
    # callBacks = [keras.callbacks.EarlyStopping(patience=2)]

    gan.fit(windows, batch_size=BATCH_SIZE, shuffle=False,)
    gan.reset_states()

    return gan
def run_gan(gan, epochs=NEPOCHS):
    data = get_data()

    if len(data) > 1:
        for epoch in range(epochs):
            printMsg ="Epoch {}/{}".format(epoch, epochs)
            if NPREV_EPOCHS_DONE:
                printMsg += " ({} done prior)".format(NPREV_EPOCHS_DONE)
            print(printMsg)

            for house in data[::-1]:
                gan = train_on_house_individually(gan, house)
    else:
        trainData = data[0].data.train
        np.random.shuffle(trainData)
        def data_gen():
            while True:
                idx = np.random.randint(0, trainData.shape[0], BATCH_SIZE)
                yield trainData[idx]

        dataGen = data_gen()

        gan.fit(dataGen, batch_size=BATCH_SIZE, shuffle=False, epochs=NEPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        gan.reset_states()

    if not gv.DEBUG:
        gan.save(genFilePath=LSTM_GENERATOR_FILE, criticFilePath=CRITIC_FILE)

    gan.plot_losses_mult_samples_epoch(
        3,
        None if gv.DEBUG else fp.folder.statefulGanImg + "W0Losses",#_CriticNoise",
        NPREV_EPOCHS_DONE
    )

    return gan

def get_synthetic_data(loadGan=True, timeStepsFactor=100, nEpochs=NEPOCHS):
    #timeStepsFactor is multiplied by the number of generator time steps (n)
    #so timeStepsFactor=10, n=16 -> 10*16 = 160 time steps on the output
    gan = get_gan(loadGan)
    if nEpochs:
        gan = run_gan(gan, epochs=nEpochs)
    genOut = genApi.get_nBatches_lstm(timeStepsFactor, gen=gan.generator, noiseDim=NOISE_DIM, batchSize=BATCH_SIZE)
    genOut = postProc.gen_out_to_real_normalized(genOut.numpy())
    return genOut

if __name__ == "__main__":
    # data = get_data()
    # common.tf_np_behavior()
    if gv.DEBUG:
        common.enable_tf_debug()
        # common.enable_tf_debug(eager=False)
        pass

    # loadGan = True
    loadGan = False
    # genOut = get_synthetic_data(loadGan, timeStepsFactor=100, nEpochs=0)

    gan = get_gan(loadGan)
    gan = run_gan(gan)

    gan.generator.reset_states()
    genOut = genApi.get_nBatches_lstm(10, gen=gan.generator, noiseDim=NOISE_DIM, batchSize=BATCH_SIZE)
    genOutProc = postProc.gen_out_one_hot_sensor(genOut.numpy())
    # genOutProc = postProc.enforce_alt_signal_each_sensor(genOutProc)

    gan.generator.reset_states()
    genOut2 = genApi.get_nBatches_lstm(10, gen=gan.generator, noiseDim=NOISE_DIM, batchSize=BATCH_SIZE)
    genOutProc2 = postProc.gen_out_one_hot_sensor(genOut2.numpy())
    # genOutProc2 = postProc.enforce_alt_signal_each_sensor(genOutProc2)

    # print(genOutProc)

    if gv.DEBUG:
        plt.show()

    # from networks.classifiers import binaryCasasClassifier as bcc
    # classifier = bcc.basic_cnn()
    # nn_diagram(classifier)
    pass