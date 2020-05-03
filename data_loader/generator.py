import numpy as np
import tensorflow.keras
import sklearn.utils
import cv2
import random
from tensorflow.keras.applications.mobilenet import preprocess_input
import time
from itertools import cycle

"""
Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, batch_size=32, dim=512, n_channels=3,
                 shuffle=False):

        """Initialization"""
        self.data = data
        self.labels = labels

        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.n_classes = 1
        self.shuffle = shuffle

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
            try:
                image_decoded = cv2.imread(frames)
                image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)

                image_decoded = cv2.resize(image_decoded, (self.dim, self.dim))
                # image_decoded = self.resize_crop(image_decoded)
                image_decoded = image_decoded.astype('float32')
                # image_decoded = preprocess_input(np.expand_dims(image_decoded, axis=0))
                # image_decoded = np.squeeze(image_decoded, axis=0)
                x[i] = image_decoded
                y[i] = labels
            except:
                continue
        # for i in range(len(x)):
        #     cv2.imshow("seq", x[i] / 255)
        #     print("Y: ", y[i])
        #     cv2.waitKey(0)
        if self.labels is not None:
            return x, y
        else:
            return x