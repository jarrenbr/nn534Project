import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import genApi
from networks import defaults


# Define the loss functions for the critic,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def critic_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
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
        latent_dim,
        critic_extra_steps=3,
        gp_weight=10.0,
    ):
        super(wgan, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = critic_extra_steps
        self.gp_weight = gp_weight
        self.history = {}

    def save(self):
        self.generator.save(self.generator.name)
        self.critic.save(self.critic.name)

    def compile(self, c_optimizer=defaults.optimizer(), g_optimizer=defaults.optimizer(),
                d_loss_fn=critic_loss, g_loss_fn=generator_loss):
        super(wgan, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the critic output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the critic and get the critic loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the critic loss
        # 6. Return the generator and critic losses as a loss dictionary

        # Train the critic first. The original paper recommends training
        # the critic for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = genApi.get_gen_out(self.generator, training=True)
                fake_logits = self.critic(fake_images, training=True)
                real_logits = self.critic(real_images, training=True)

                # Calculate the critic loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original critic loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the critic loss
            d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.c_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables)
            )

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = genApi.get_gen_out(self.generator, training=True,)
            # Get the critic logits for fake images
            gen_img_logits = self.critic(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {wgan.criticLossKey: d_loss, wgan.genLossKey: g_loss}

    def fit(self, *args, **kwargs):
        self.history = super(wgan, self).fit(*args, **kwargs)
        return self.history

    def plot_losses(self):
        plt.plot(self.history.history[wgan.criticLossKey])
        plt.plot(self.history.history[wgan.genLossKey])
        plt.hlines(0, 0, len(self.history.history[wgan.criticLossKey]), 'k', 'dashed')
        plt.title("WGAN Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend([wgan.criticLossKey, wgan.genLossKey])

    def call(self, input):
        #only for keras' validation
        return self.critic(input)

    def reset_states(self):
        self.generator.reset_states()
        self.critic.reset_states()


if __name__ == "__main__":
    gan = wgan()

    pass