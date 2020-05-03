import pandas as pd
import numpy as np
import os
import sklearn.utils
import matplotlib
import matplotlib.pyplot as plt
import random

from PIL import Image
from random import shuffle

import tensorflow as tf
import tensorflow.keras.backend as K

import utils
from data_loader.generator import DataGenerator
from model.regression import RegressionModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
utils.seed_everything()



PATH = "D:\\Data\\alaska2-image-steganalysis\\"
IMG_SIZE = 512
train_val_ratio = 0.7

# Cover Images
cover_ids = os.listdir(os.path.join(PATH, 'Cover'))
cover_labels = [-1] * len(cover_ids)
for i in range(len(cover_ids)):
    cover_ids[i] = os.path.join(os.path.join(PATH, 'Cover'), cover_ids[i])

# Crypted Images
JMiPOD_ids = os.listdir(os.path.join(PATH, 'JMiPOD'))
for i in range(len(JMiPOD_ids)):
    JMiPOD_ids[i] = os.path.join(os.path.join(PATH, 'JMiPOD'), JMiPOD_ids[i])
JUNIWARD_ids = os.listdir(os.path.join(PATH, 'JUNIWARD'))
for i in range(len(JUNIWARD_ids)):
    JUNIWARD_ids[i] = os.path.join(os.path.join(PATH, 'JUNIWARD'), JUNIWARD_ids[i])
UERD_ids = os.listdir(os.path.join(PATH, 'UERD'))
for i in range(len(UERD_ids)):
    UERD_ids[i] = os.path.join(os.path.join(PATH, 'UERD'), UERD_ids[i])

crypt_ids = JMiPOD_ids + JUNIWARD_ids + UERD_ids
crypt_labels = [1] * len(crypt_ids)
N_IMAGES = len(cover_ids)*train_val_ratio
print("Number of images:"
      "\n\t Cover: {} \n\t JMiPOD: {} \n\t JUNIWARD: {} \n\t UERD: {}".format(len(cover_ids),
                                                                              len(JMiPOD_ids),
                                                                              len(JUNIWARD_ids),
                                                                              len(UERD_ids)))
n_samples = int(len(cover_labels)*train_val_ratio)
n_samples_val = len(cover_labels) - n_samples

print("Splitting the dataset:\n\t - Training: \n\t\t Cover: {} \n\t\t Crypt {} \n\t - Validation: "
      "\n\t\t Cover: {} \n\t\t Crypt {}".format(n_samples, n_samples*3, n_samples_val, n_samples_val*3))

cover_ids = sklearn.utils.shuffle(cover_ids)
crypt_ids = sklearn.utils.shuffle(crypt_ids)
IMAGE_IDS_train = cover_ids[:n_samples] + crypt_ids[:n_samples*3]
IMAGE_LABELS_train = cover_labels[:n_samples] + crypt_labels[:n_samples*3]

IMAGE_IDS_val = []
IMAGE_LABELS_val = []
IMAGE_IDS_val = cover_ids[-n_samples_val:] + crypt_ids[-n_samples_val*3:]
IMAGE_LABELS_val = cover_labels[-n_samples_val:] + crypt_labels[-n_samples_val*3:]

sample_sub = pd.read_csv(PATH + 'sample_submission.csv')


train_gen = DataGenerator(IMAGE_IDS_train, IMAGE_LABELS_train, batch_size=4, shuffle=True)
validation_gen = DataGenerator(IMAGE_IDS_val, IMAGE_LABELS_val, batch_size=4)

print("Loading model")
# TODO: Decide if use preprocess
regression = RegressionModel()
model = regression.build_model()
model.summary()

model.fit(x=train_gen,
          steps_per_epoch=len(train_gen),
          validation_data=validation_gen,
          validation_steps=len(validation_gen),
          epochs=10,
          callbacks=regression.callbacks)
