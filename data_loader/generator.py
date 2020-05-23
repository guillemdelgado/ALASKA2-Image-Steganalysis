import numpy as np
import tensorflow.keras
import sklearn.utils
import cv2
import jpegio as jio
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import itertools
import random
from utils.data_augmentation import horizontal_flip, vertical_flip, rotation, TTA
from utils.utils import JPEGdecompressYCbCr
from tensorflow.keras.applications.mobilenet import preprocess_input
import time
from itertools import cycle
import matplotlib.pyplot as plt

"""
Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, batch_size=32, dim=512, n_channels=3,
                 shuffle=False, sampling=None, data_augmentation=False, format="RGB", tta=None):

        """Initialization"""
        if sampling == 'under_sample':
            rus = RandomUnderSampler(random_state=0, replacement=True)
            X_resampled, y_resampled = rus.fit_resample(np.array(data).reshape(-1, 1), np.array(labels))
            self.data = list(itertools.chain(*X_resampled.tolist()))
            self.labels = y_resampled.tolist()
        elif sampling == 'over_sample':
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = ros.fit_resample(np.array(data).reshape(-1, 1), np.array(labels))
            self.data = list(itertools.chain(*X_resampled.tolist()))
            self.labels = y_resampled.tolist()
        else:
            self.data = data
            self.labels = labels

        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.n_classes = 1
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.format = format
        self.tta = tta
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        if self.labels is not None:
            x, y = self.__data_generation(indexes)
            return x, y
        else:
            x = self.__data_generation(indexes)
            return x

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))

        if self.shuffle:
            self.indexes = sklearn.utils.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size, self.dim, self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        # Generate data

        list_IDs_temp = [self.data[k] for k in indexes]

        if self.labels is not None:
            list_labels_temp = [self.labels[k] for k in indexes]
        else:
            list_labels_temp = [None for k in indexes]

        for i, data in enumerate(zip(list_IDs_temp, list_labels_temp)):
            frames = data[0]
            labels = data[1]
            if self.format == "RGB":
                image_decoded = cv2.imread(frames)
                image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
            elif self.format == "YCBCR":
                jpegStruct = jio.read(frames)
                image_decoded = JPEGdecompressYCbCr(jpegStruct)
            elif self.format == "DCT":
                jpegStruct = jio.read(frames)
                coverDCT = np.zeros([512, 512, 3])
                coverDCT[:, :, 0] = jpegStruct.coef_arrays[0]
                coverDCT[:, :, 1] = jpegStruct.coef_arrays[1]
                coverDCT[:, :, 2] = jpegStruct.coef_arrays[2]
                image_decoded = coverDCT

            image_decoded = cv2.resize(image_decoded, (self.dim, self.dim))

            if self.data_augmentation and self.format != "DCT":
                if random.uniform(0, 1):
                    image_decoded = horizontal_flip(image_decoded)
                if random.uniform(0, 1):
                    image_decoded = vertical_flip(image_decoded)
                if random.uniform(0, 1):
                    image_decoded = rotation(image_decoded)

            if self.tta is not None:
                image_decoded = TTA(image_decoded, self.tta)
            # image_decoded = self.resize_crop(image_decoded)
            image_decoded = image_decoded.astype('float32')
            # image_decoded = preprocess_input(np.expand_dims(image_decoded, axis=0))
            # image_decoded = np.squeeze(image_decoded, axis=0)
            x[i] = image_decoded
            if self.labels is not None:
                y[i] = labels
        if self.labels is not None:
            return x, y
        else:
            return x
