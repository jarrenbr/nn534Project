import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import files.filePaths as fp
from networks.gans import genApi

MODEL_DIR = fp.folder.kmModel + "statefulGan/"
# def classifier_naming

GENERATOR_FILE = MODEL_DIR + "generator" + fp.extensions.kerasModel
CRITIC_FILE = MODEL_DIR + "critic" + fp.extensions.kerasModel

GENERATOR_BATCH_SIZE = 16
CRITIC_BATCH_SIZE = 128



if __name__ == "__main__":
    gan = wgan.wgan()
    exit()