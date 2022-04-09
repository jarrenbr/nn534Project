import keras.models
from keras.layers import BatchNormalization, Dense, Reshape, Flatten, Conv1D, Concatenate
from keras.layers import Conv2DTranspose, LeakyReLU, Dropout, Embedding, Activation
from processData.binaryCasasProcess import binaryCasasData as bcData
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randn, randint
from keras.models import Model, Input
from utils import globalVars as gv
import matplotlib.pyplot as plt
from matplotlib import pyplot
import einops as einops
from numpy import zeros
from numpy import ones
import pandas as pd
import numpy as np
import tensorflow
import csv

real_data_loss = []
fake_data_loss = []
real_data_acc = []
fake_data_acc = []


# define the standalone discriminator model
def define_discriminator(in_shape=(384, 48), n_classes=14):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)

    # down sample to 14x14
    fe = Conv1D(24, 3, strides=2, padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # normal
    fe = Conv1D(48, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # down sample to 7x7
    fe = Conv1D(64, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # down sample one more
    fe = Conv1D(192, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(48, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(48, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2])
    # compile model
    opt = tensorflow.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=opt, metrics=['accuracy'])
    model.summary()
    # plot the model
    tensorflow.keras.utils.plot_model(model, to_file='results/discriminator_plot.png', show_shapes=True,
                                      show_layer_names=True)
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=14):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    depth = 48  # 32
    ks = 3
    dropout = 0.25
    dim = 96  #
    # label input
    in_label = Input(shape=(latent_dim,))
    # embedding for categorical input
    li = Embedding(150, 1)(in_label)
    # linear multiplication
    n_nodes = dim * 1
    li = Dense(n_nodes)(li)

    # reshape to additional channel
    li = Reshape((dim, 1, 48))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = dim * depth
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, 1, depth))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])  # gen=96,1,32 x li=96,1,1
    # up sample to 192,1,16
    gen = Conv2DTranspose(96, 3, strides=(2, 1), padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # up sample to  384,1,8
    gen = Conv2DTranspose(48, 3, strides=(2, 1), padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # up sample
    # gen = Conv2DTranspose(48, (3,3), strides=(2,1), padding='same', kernel_initializer=init)(gen)
    # gen = BatchNormalization()(gen)
    # gen = Activation('relu')(gen)
    # 384 x 1 property image
    gen = Reshape((384, 48))(gen)
    # up sample to 28x28
    # gen = Conv1DTranspose(1, 3, padding='same', kernel_initializer=init)(gen)
    gen = Conv1D(48, 3, strides=1, padding='same')(gen)
    out_layer = Activation('relu')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    model.summary()
    # plot the model
    tensorflow.keras.utils.plot_model(model, to_file='results/generator_plot.png', show_shapes=True,
                                      show_layer_names=True)
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
    model.compile(loss=['binary_crossentropy'], optimizer=opt, metrics=['accuracy'])
    # summarise the model
    model.summary()
    # plot the model
    tensorflow.keras.utils.plot_model(model, to_file='results/gan_plot.png', show_shapes=True, show_layer_names=True)
    return model


# load images
def load_real_samples():
    df1 = pd.read_csv('data/binaryCasas/processed/b1Train.csv', skiprows=1)
    df2 = pd.read_csv('data/binaryCasas/processed/b2Train.csv', skiprows=1)
    df3 = pd.read_csv('data/binaryCasas/processed/b3Train.csv', skiprows=1)
    df = pd.concat([df1, df2, df3])
    df = df.drop(df.tail(45614).index)
    # load dataset
    data_xy = df.astype('float')
    pd.DataFrame(data_xy)
    scaler = MinMaxScaler(copy=False)
    window = 384
    n = ((np.where(np.any(data_xy, axis=1))[0][-1] + 1) // window) * window
    xx = (scaler.fit_transform(data_xy.iloc[:n, 0].values.reshape(-1, 1))).reshape(28176, 48)
    y_train = (data_xy.iloc[:(n - window), 1].values.reshape(-1, 1)).reshape(28168, 48)

    # make to matrix
    x_train = np.asarray([xx[i:i + window] for i in range(28168 - window)], dtype=float)

    X = np.asarray(x_train.copy())
    train_y = np.asarray(y_train.copy())

    X = (X - 127.5) / 127.5
    print(X.shape, train_y.shape)
    return [X, train_y]


# select real samples
def generate_real_samples(dataset, n_samples=28168):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0]//128, n_samples)
    # select images and labels
    X, labels = np.asarray(images[ix]), np.asarray(labels[ix])
    # generate class labels
    y = ones((n_samples, 48))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=48):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, (n_samples, 48))  # check these labels!
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples=1):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim,  n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 48))
    return [images, labels_input], y


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, nmn_label], nmn_y = generate_fake_samples(g_model, latent_dim,
                                                  n_samples)
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
        # np.savetxt('results/test_raw{}_{}.csv'.format(i, step), X[i], delimiter=',')
        # np.savetxt('test_cat{}_{}.csv'.format(i, step), nmn_label[i], delimiter=',')
    # save plot to file
    # np.savetxt('results/test_raw_nc{}.csv'.format(step), X[:,:,0], delimiter=',')
    # np.savetxt('test_cat_nc{}.csv'.format(step), nmn_label[:],delimiter=',')
    filename1 = 'generated_plot_%04d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=5, n_batch=4):
    # calculate the number of batches per training epoch
    bat_per_epo = 10 # 50 # 100 # int(dataset[0].shape[0] / n_batch)
    print('batch per epoch: %d' % bat_per_epo)
    # calculate the number of training iterations
    n_steps = 10 # 500 # 1000 # bat_per_epo * n_epochs # 15
    print('number of steps: %d' % n_steps)
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset)
        # update discriminator model weights
        _, d_r1, d_r2, _, _ = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim)
        # write_data to file
        write_synthetic_to_csv(X_fake, i)
        # evaluate the model
        # update discriminator model weights
        _, d_f, d_f2, _, _ = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # evaluate
        loss_real, _, _, acc_real, _ = d_model.evaluate(X_real, labels_real, verbose=0)
        loss_fake, _, _, acc_fake, _ = d_model.evaluate(X_fake, y_fake, verbose=0)
        real_data_loss.append(loss_real)
        real_data_acc.append(acc_real)
        fake_data_loss.append(loss_fake)
        fake_data_acc.append(acc_fake)

        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 48))
        # update the generator via the discriminator's error
        loss, g_1, g_2, acc = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
        # evaluate the model performance every 'epoch'
        # if (i + 1) % bat_per_epo == 0:
        #     summarize_performance(i, g_model, latent_dim)


def write_synthetic_to_csv(x, i):
    with open('results/synthetic_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['Time', 'Signal', 'D021', 'D022', 'D023', 'D024', 'D025', 'D026', 'D027', 'D028', 'D029', 'D030',
                  'D031', 'D032', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010',
                  'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'Bathing',
                  'Bed_Toilet_Transition', 'Eating', 'Enter_Home', 'Housekeeping', 'Leave_Hom', 'Meal_Preparation',
                  'Other_Activity', 'Personal_Hygiene', 'Relax', 'Sleeping_Not_in_Bed', 'Sleeping_in_Bed',
                  'Take_Medicine', 'Work']
        # start with (96, 384, 48) to (96*384, 48):
        arr = einops.rearrange(x, 'h w i -> (h w) i')
        # write the header
        writer.writerow(header)
        row = []
        for w in range(384):
            row = arr[w].flatten()
            writer.writerow(row)
        f.close()


def main():
    # size of the latent space
    latent_dim = 48
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

    from utils import filePaths as fp, globalVars as gv
    if not gv.DEBUG:
        gan_model.save(fp.folder.kmModel + "baseGan.km")

    #next time
    #gan_model = keras.models.load_model(fp.folder.kmModel + "baseGan.km")

    blah_a = [x for x in range(len(real_data_loss))]

    plt.plot(blah_a, real_data_loss, label="real data loss", color='blue', )
    plt.plot(blah_a, fake_data_loss, label="fake data loss", color='red', )
    plt.xlabel('Iteration number')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.title('Graph that Shows Error by Iteration')
    plt.savefig('results/Error_Comparison.png')
    plt.show()
    plt.clf()

    plt.plot(blah_a, real_data_acc, label="real data acc", color='purple', )
    plt.plot(blah_a, fake_data_acc, label="fake data acc", color='green', )
    plt.xlabel('Iteration number')
    plt.ylabel('Accuracy Rate')
    plt.legend()
    plt.title('Graph that Shows Accuracy by Iteration')
    plt.savefig('results/Acc_Comparison.png')
    plt.show()


if __name__ == "__main__":
    main()