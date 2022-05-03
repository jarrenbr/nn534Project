import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from networks import defaults, commonBlocks as cBlocks
from networks.gans import genApi
from processData.binaryCasasProcess import postProcess as postProc, binaryCasasData as bcData
from names import binaryCasasNames as bcNames
from utils import globalVars as gv, filePaths as fp, common

DISC_NAME = "baseGanDisc"
GEN_NAME = "baseGanGen"
KM_FOLDER = fp.folder.kmModel + "baseGan/"
DISC_FILE = KM_FOLDER + DISC_NAME
GEN_FILE = KM_FOLDER + GEN_NAME

NTIMESTEPS = 16
NOISE_DIM = defaults.NOISE_DIM
BATCH_SIZE = defaults.BATCH_SIZE
NEPOCHS = 2 if gv.DEBUG else 10
NPREV_EPOCHS_DONE = 0

def get_discriminator(nTimesteps=NTIMESTEPS)->keras.Model:
    inputLayer = keras.Input(shape=(nTimesteps, bcNames.nGanFeatures))
    x = layers.GaussianNoise(.025)(inputLayer)
    x = layers.Dense(bcNames.nGanFeatures, defaults.leaky_relu())(x)

    args = [
        cBlocks.conv_args(nFilters=150, kernelSize=2),
        cBlocks.conv_args(nFilters=150, kernelSize=2),
        cBlocks.conv_args(nFilters=160, kernelSize=2),
        ]
    for arg in args:
        x = layers.Conv1D(**arg.kwargs)(x)
        x = cBlocks.block(x, activation=defaults.leaky_relu(), use_bn=True)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation=keras.activations.sigmoid)(x)

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= DISC_NAME)

    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    # print(model.summary())
    return model

def get_generator(noiseDim=NOISE_DIM) -> keras.models.Model:
    #goal: 16 X 48
    inputLayer = keras.Input(
        shape=(1,noiseDim,)
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

    time = layers.Dense(1, keras.activations.sigmoid)(x)
    signal = layers.Dense(1, keras.activations.sigmoid)(x)
    sensors = layers.Dense(len(bcNames.allSensors), keras.activations.softmax)(x)
    activities = layers.Dense(bcNames.nLabels, keras.activations.softmax)(x)
    x = layers.Concatenate()([time, signal, sensors, activities])

    model = keras.models.Model(inputs=[inputLayer], outputs=[x], name= GEN_NAME)
    # model.compile(loss = keras.losses.CategoricalCrossentropy(),
    #               optimizer = defaults.optimizer(), metrics = defaults.METRICS)
    # print(model.summary())
    return model

def common_loss():
    return keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=.2)

def discriminator_loss(realLogits, fakeLogits):
    lossFn = common_loss()
    ones = tf.ones(shape=realLogits.shape)
    zeros = tf.zeros(shape=fakeLogits.shape)
    realLoss = lossFn(ones, realLogits)
    fakeLoss = lossFn(zeros, fakeLogits)
    return (realLoss + fakeLoss) / 2.

def generator_loss(fakeLogits):
    lossFn = common_loss()
    ones = tf.ones(shape=fakeLogits.shape)
    fakeLoss = lossFn(ones, fakeLogits)
    return fakeLoss

class gan(keras.Model):
    discLossKey = "discLoss"
    genLossKey = "genLoss"

    def __init__(self, discriminator, generator, latentDim=NOISE_DIM, nTimesteps=NTIMESTEPS, nDiscSteps=1, batchSize = BATCH_SIZE):
        super(gan, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latentDim = latentDim
        self.nTimesteps = nTimesteps
        self.nDiscSteps = nDiscSteps
        self.batchSize = batchSize

    def compile(self, dOptimizer=defaults.optimizer(), gOptimizer=defaults.optimizer(),
                d_loss_fn=discriminator_loss, g_loss_fn=generator_loss):
        super(gan, self).compile()
        self.dOptimizer = dOptimizer
        self.gOptimizer = gOptimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, realData):
        if isinstance(realData, tuple):
            realData = realData[0]


        for dStep in range(self.nDiscSteps):
            genData = genApi.get_gen_out(self.generator, noiseDim=self.latentDim, batchSize=self.batchSize)
            with tf.GradientTape() as tape:
                fakeLogits = self.discriminator(genData)
                realLogits = self.discriminator(realData)
                dLoss = self.d_loss_fn(realLogits, fakeLogits)
            dGradient = tape.gradient(dLoss, self.discriminator.trainable_variables)
            self.dOptimizer.apply_gradients(zip(dGradient, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            genData = genApi.get_gen_out(self.generator, noiseDim=self.latentDim, batchSize=self.batchSize)
            fakeLogits = self.discriminator(genData)
            gLoss = self.g_loss_fn(fakeLogits)

        gGradient = tape.gradient(gLoss, self.generator.trainable_variables)
        self.gOptimizer.apply_gradients(zip(gGradient, self.generator.trainable_variables))

        return {gan.discLossKey: dLoss, gan.genLossKey : gLoss}


    def save(self, genFilePath, criticFilePath):
        self.generator.save(genFilePath)
        self.discriminator.save(criticFilePath)

def get_gan(loadGan=False):
    if loadGan:
        gen = keras.models.load_model(GEN_FILE)
        disc = keras.models.load_model(DISC_FILE)
    else:
        gen = get_generator()
        disc = get_discriminator()

    baseGan = gan(
        disc, gen, defaults.NOISE_DIM, batchSize=BATCH_SIZE,
    )
    baseGan.compile()
    return baseGan

def get_data(batchSize = BATCH_SIZE):
    return bcData.get_all_homes_as_xy_combined_gen(
        batchSize, NTIMESTEPS, firstN=gv.DATA_AMT)

def train_on_house(baseGan, house):
    windows = house.data.train.data

    nOmitted = (windows.shape[0] % (BATCH_SIZE * NTIMESTEPS))
    validSize = windows.shape[0] - nOmitted
    # print("House {}. Used:Omitted = {}:{}".format(house.name, validSize, nOmitted))
    assert validSize > 0

    # random offset if some are omitted
    # randomly choose to omit head or tail
    if nOmitted:
        offset = np.random.randint(0, min(NTIMESTEPS, nOmitted))
        windows = windows[nOmitted-offset:-offset] if np.random.randint(0,2) else windows[offset:validSize+offset]
    else:
        windows = windows[nOmitted:] if np.random.randint(0,2) else windows[:validSize]

    windows = np.reshape(
        windows,
        (-1, NTIMESTEPS, bcNames.nGanFeatures)
    )
    baseGan.fit(windows, batch_size=BATCH_SIZE, shuffle=True,)
    return baseGan

def train_gan(baseGan, epochs=NEPOCHS):
    data = get_data()

    for epoch in range(epochs):
        printMsg ="Epoch {}/{}".format(epoch, epochs)
        if NPREV_EPOCHS_DONE:
            printMsg += " ({} done prior)".format(NPREV_EPOCHS_DONE)
        print(printMsg)

        for house in data[::-1]:
            baseGan = train_on_house(baseGan, house)

    if not gv.DEBUG:
        baseGan.save(genFilePath=GEN_FILE, criticFilePath=DISC_FILE)

    # baseGan.plot_losses_mult_samples_epoch(
    #     3, None if gv.DEBUG else fp.folder.basicGanImg + "Losses", NPREV_EPOCHS_DONE
    # )

    return baseGan

if __name__ == "__main__":
    if gv.DEBUG:
        common.enable_tf_debug()
    # loadGan = True
    loadGan = False
    baseGan = get_gan(loadGan)
    baseGan = train_gan(baseGan)


    exit()
