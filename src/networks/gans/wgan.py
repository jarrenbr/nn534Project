import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import genApi
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
    homesPrefix = ["Home%d" %(i,) for i in range(1,4)]

    def __init__(
        self,
        critic,
        generator,
        latentDim,
        nCriticTimesteps,
        nGenTimesteps,
        batchSize = defaults.BATCH_SIZE,
        criticExtraSteps=1,
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
        self.fullHistory = tf.keras.callbacks.History()

    def save(self, genFilePath, criticFilePath):
        self.generator.save(genFilePath)
        self.critic.save(criticFilePath)

    def compile(self, c_optimizer=defaults.optimizer(), g_optimizer=defaults.optimizer(),
                c_loss_fn=critic_loss, g_loss_fn=generator_loss):
        super(wgan, self).compile()
        self.c_optimizer = c_optimizer
        self.gOptimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn


    def get_gen_out_for_critic(self):
        fakeImgs = []
        for _ in range(self.genToCriticFactor):
            fakeImgs.append(
                genApi.get_gen_out(self.generator, training=True, batchSize=self.batchSize)
            )
        return tf.concat(fakeImgs, axis=1)

    def gradient_penalty(self, realImgs, fakeImgs):
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
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gpTape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, realImgs,):
        if isinstance(realImgs, tuple):
            realImgs = realImgs[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the critic and get the critic loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the critic loss
        # 6. Return the generator and critic losses as a loss dictionary

        #cannot recreate the synthetic data multiple times else it will confuse the generator (in theory)
        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            fakeImgs = self.get_gen_out_for_critic()
            # Get the critic logits for fake images
            genImgLogits = self.critic(fakeImgs, training=True)
            # Calculate the generator loss
            gLoss = self.g_loss_fn(genImgLogits)

        # Get the gradients w.r.t the generator loss
        genGradient = tape.gradient(gLoss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.gOptimizer.apply_gradients(
            zip(genGradient, self.generator.trainable_variables)
        )

        for i in range(self.cSteps):
            with tf.GradientTape() as tape:
                fakeLogits = self.critic(fakeImgs, training=True)
                realLogits = self.critic(realImgs, training=True)

                # Calculate the critic loss using the fake and real image logits
                cCost = self.c_loss_fn(real_img=realLogits, fake_img=fakeLogits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(realImgs, fakeImgs)
                # Add the gradient penalty to the original critic loss
                cLoss = cCost + gp * self.gpWeight

            # Get the gradients w.r.t the critic loss
            cGradient = tape.gradient(cLoss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.c_optimizer.apply_gradients(
                zip(cGradient, self.critic.trainable_variables)
            )

        return {wgan.criticLossKey: cLoss, wgan.genLossKey: gLoss}

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
