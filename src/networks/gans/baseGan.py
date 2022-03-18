import pandas as pd
from keras.layers import BatchNormalization, Dense, Reshape,  Flatten, Conv1D, Concatenate
from keras.layers import Conv2DTranspose, LeakyReLU, Dropout, Embedding, Activation
from sklearn.preprocessing import MinMaxScaler
from names import binaryCasasNames as names
from sklearn.impute import SimpleImputer
from numpy.random import randn, randint
from keras.models import Model, Input
from matplotlib import pyplot
from numpy import zeros
from numpy import ones
import numpy as np
import tensorflow
import keras

# define the standalone discriminator model
def define_discriminator(in_shape=(384, 1), n_classes=4):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # downsample to 14x14
    fe = Conv1D(16, 3, strides=2, padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    # normal
    fe = Conv1D(32, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    # downsample to 7x7
    fe = Conv1D(64, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # downsample one more
    fe = Conv1D(128, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = tensorflow.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=4):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    depth = 32  # 32
    ks = 3
    dropout = 0.25
    dim = 96  #
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 96 * 1
    li = Dense(n_nodes)(li)

    # reshape to additional channel
    li = Reshape((96, 1, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = dim * depth
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, 1, depth))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])  # gen=96,1,32 x li=96,1,1
    # upsample to 192,1,16
    gen = Conv2DTranspose(16, 3, strides=(2, 1), padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # upsample to  384,1,8
    gen = Conv2DTranspose(8, 3, strides=(2, 1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # updamsple
    # gen = Conv2DTranspose(48, (3,3), strides=(2,1), padding='same', kernel_initializer=init)(gen)
    # gen = BatchNormalization()(gen)
    # gen = Activation('relu')(gen)
    # 384 x 1 property image
    gen = Reshape((384, -1))(gen)
    # upsample to 28x28
    # gen = Conv1DTranspose(1, 3, padding='same', kernel_initializer=init)(gen)
    gen = Conv1D(1, 3, strides=1, padding='same')(gen)
    out_layer = Activation('tanh')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    model.summary()
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = tensorflow.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# load images
def load_real_samples():
    df1 = pd.read_csv('data/binaryCasas/processed/b1Test.csv', skiprows=1)
    # load dataset
    dataxy = df1.astype('int')
    pd.DataFrame(dataxy)
    scaler = MinMaxScaler(copy=False)
    window = 384
    n = ((np.where(np.any(dataxy, axis=1))[0][-1] + 1) // window) * window
    xx = scaler.fit_transform(dataxy.iloc[:n, 0].values.reshape(-1, 1))
    y_train = dataxy.iloc[:(n - window), 1].values.reshape(-1, 1)

    # make to matrix
    x_train = np.asarray([xx[i:i + window] for i in range(n - window)])

    X = x_train.copy()
    trainy = y_train.copy()

    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=4):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)  # check these labels!
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, nmn_label], nmn_y = generate_fake_samples(g_model, latent_dim,
                                                  n_samples)  # TODO!:Numan (nmns were _ and _) - change labels in this row and debug!
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :], cmap='gray_r')
        np.savetxt('%s.csv' % ("test_raw",) % (i, step), X[i, :], delimiter=',')
        np.savetxt('%s.csv' % ("test_cat",) % (i, step), nmn_label[i], delimiter=',')
    # save plot to file
    # np.savetxt('test_raw_nc%d.csv' % (step), X[:,:,0], delimiter=',')
    # np.savetxt('test_cat_nc%d.csv' % (step), nmn_label[:],delimiter=',')
    filename1 = 'generated_plot_%04d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=30, n_batch=64):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print('batch per epoch: %d' % bat_per_epo)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print('number of steps: %d' % n_steps)
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
        # evaluate the model performance every 'epoch'
        if (i + 1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, latent_dim)


def main():
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(generator, discriminator)
    # load image data
    dataset = load_real_samples()
    # train model
    train(generator, discriminator, gan_model, dataset, latent_dim)

if __name__ == "__main__":
    main()
