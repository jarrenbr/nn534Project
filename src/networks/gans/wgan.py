import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from networks.gans import genApi
from networks import defaults

# We will add the gradient penalty later to this loss function.
def critic_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


class wgan(keras.Model):
    criticLossKey = "cLoss"
    genLossKey = "gLoss"
    realHelperKey = "rhLoss"
    homesPrefix = ["Home%d" %(i,) for i in range(1,4)]

    def __init__(
        self,
        critic,
        generator,
        latentDim,
        nCriticTimesteps,
        nGenTimesteps,
        batchSize = defaults.BATCH_SIZE,
        criticExtraSteps=10,
        gpWeight=10.0,
    ):
        super(wgan, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latentDim

        assert nCriticTimesteps % nGenTimesteps == 0
        self.genToCriticFactor = int(nCriticTimesteps / nGenTimesteps)

        self.batchSize = batchSize
        self.cSteps = criticExtraSteps
        self.gpWeight = gpWeight

        self.realHelper = self.create_lstm_helper((batchSize, *self.critic.input_shape[0][-2:]), "realHelper")
        # self.fakeHelper = self.create_lstm_helper(self.critic.input_shape, "fakeHelper")
        self.fullHistory = tf.keras.callbacks.History()

    def save(self, genFilePath, criticFilePath):
        self.generator.save(genFilePath)
        self.critic.save(criticFilePath)

    def compile(self, c_optimizer=defaults.optimizer(), g_optimizer=defaults.optimizer(),
                c_loss_fn=critic_loss, g_loss_fn=generator_loss, *args, **kwargs):
        super(wgan, self).compile(*args, **kwargs)
        self.c_optimizer = c_optimizer
        self.gOptimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.rh_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn


    def create_lstm_helper(self, inputShape, name)->keras.models.Model:
        inputLayer = keras.Input(batch_shape=inputShape)
        x = layers.Bidirectional(
            layers.LSTM(
                inputShape[-1],
                stateful=True,
                return_sequences=True
            )
        )(inputLayer)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(inputShape[-1], activation=keras.activations.sigmoid)(x)

        model = keras.models.Model(inputs=[inputLayer], outputs=[x], name=name)
        return model

    def helper_gp(self, realImgs):
        # Get the interpolated image
        alpha = tf.random.normal([self.batchSize, 1, 1], 0.0, .1)
        distorted = realImgs + alpha

        with tf.GradientTape() as gpTape:
            gpTape.watch(distorted)
            # 1. Get the critic output for this interpolated image.
            pred = self.realHelper(distorted, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gpTape.gradient(pred, [distorted])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def gradient_penalty_rh(self, realImgs, fakeImgs):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([self.batchSize, 1, 1], 0.0, 1.0)
        diff = fakeImgs - realImgs
        interpolated = realImgs + alpha * diff

        with tf.GradientTape() as gpTape:
            gpTape.watch(interpolated)
            # 1. Get the critic output for this interpolated image.
            pred = self.realHelper([interpolated], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gpTape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    def gradient_penalty_context(self, realImgs, fakeImgs, realContext, fakeContext):
        # Get the interpolated image
        alpha = tf.random.normal([self.batchSize, 1, 1], 0.0, 1.0)
        diff = fakeImgs - realImgs
        interpolated = realImgs + alpha * diff

        alpha2 = tf.random.normal([self.batchSize, 1, 1], 0.0, 1.0)
        diff2 = fakeContext - realContext
        contextInterpolated = realContext + alpha2 * diff2


        with tf.GradientTape() as gpTape:
            gpTape.watch(interpolated)
            gpTape.watch(contextInterpolated)
            # 1. Get the critic output for this interpolated image.
            pred = self.critic([interpolated, contextInterpolated], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gpTape.gradient(pred, [interpolated, contextInterpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, realImgs,):
        if isinstance(realImgs, tuple):
            realImgs = realImgs[0]


        #cannot recreate the synthetic data multiple times else it will confuse the generator (in theory)
        # Train the generator
        with tf.GradientTape() as tape:
            fakeImgs, genContext = self.generator(genApi.get_gen_input(shape=(self.batchSize,1, self.latent_dim)))
            genImgLogits = self.critic([fakeImgs, genContext], training=True)
            gLoss = self.g_loss_fn(genImgLogits)


        # Get the gradients w.r.t the generator loss
        genGradient = tape.gradient(gLoss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.gOptimizer.apply_gradients(
            zip(genGradient, self.generator.trainable_variables)
        )

        for i in range(self.cSteps):
            #train context
            with tf.GradientTape() as tape:
                realContext = self.realHelper(realImgs)
                predGenContext = self.realHelper(fakeImgs)
                rhCost =  self.c_loss_fn(real_img=genContext, fake_img=predGenContext)
                rhgp = self.gradient_penalty_rh(realImgs=genContext, fakeImgs=predGenContext)
                rhLoss = rhCost + rhgp * self.gpWeight

            rhGradient = tape.gradient(rhLoss, self.realHelper.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(rhGradient, self.realHelper.trainable_variables)
            )

            #train critic
            with tf.GradientTape() as tape:
                realLogits = self.critic([realImgs, realContext], training=True)
                cCost = self.c_loss_fn(real_img=realLogits, fake_img=genImgLogits)
                cgp = self.gradient_penalty_context(realImgs, fakeImgs, realContext, genContext)
                cLoss = cCost + cgp * self.gpWeight

            # Get the gradients w.r.t the critic loss
            cGradient = tape.gradient(cLoss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.c_optimizer.apply_gradients(
                zip(cGradient, self.critic.trainable_variables)
            )


        return {wgan.criticLossKey: cLoss, wgan.genLossKey: gLoss, wgan.realHelperKey : rhLoss}

    def fit(self, *args, **kwargs):
        super(wgan, self).fit(*args, **kwargs)
        if len(self.fullHistory.history) == 0:
            self.fullHistory = self.history
            #future aggregations make these misleading
            self.fullHistory.params = None
            self.fullHistory.epoch = None
        else:
            for key, value in self.history.history.items():
                self.fullHistory.history[key] += value

        return self.history

    def plot_losses(self, saveFile=None,xs=None):
        if xs is None:
            xs = [i for i in range(len(self.fullHistory.history[wgan.criticLossKey]))]
        plt.plot(xs, self.fullHistory.history[wgan.criticLossKey])
        plt.plot(xs, self.fullHistory.history[wgan.genLossKey])
        plt.hlines(0, xs[0], xs[0] + len(self.fullHistory.history[wgan.criticLossKey]), 'k', 'dashed')
        plt.title("WGAN Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend([wgan.criticLossKey, wgan.genLossKey])
        if saveFile:
            plt.savefig(saveFile)

    def plot_losses_mult_samples_epoch(self, samplesPerEpoch, saveFile=None, nEpochsPrior=0, ):
        nYs = len(self.fullHistory.history[wgan.criticLossKey])
        nEpochs = nYs / samplesPerEpoch
        xs = np.linspace(nEpochsPrior + 1/samplesPerEpoch, nEpochsPrior + nEpochs, num=nYs)
        plt.plot(xs, self.fullHistory.history[wgan.criticLossKey])
        plt.plot(xs, self.fullHistory.history[wgan.genLossKey])
        plt.hlines(0, xs[0], xs[-1], 'k', 'dashed')
        plt.title("WGAN Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend([wgan.criticLossKey, wgan.genLossKey])
        if saveFile:
            plt.savefig(saveFile)


    def call(self, input):
        #only for keras' validation
        return self.critic(input)

    def reset_states(self):
        self.generator.reset_states()
        # self.critic.reset_states()


if __name__ == "__main__":
    gan = wgan()

    exit()
