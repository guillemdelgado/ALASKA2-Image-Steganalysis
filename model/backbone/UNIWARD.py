import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import os
from model.layers.layers import ReflectionPadding2D



class UNIWARD:
    def __init__(self, input_tensor, pooling='avg'):
        self.input_tensor = input_tensor

        x = self.conv_block(input_tensor, filters=8, kernel_size=3, strides=1, activation='relu', use_dropout=True)
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='same')(x)
        x = self.conv_block(x, filters=16, kernel_size=3, strides=1, activation='relu', use_dropout=True)
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='same')(x)
        x = self.conv_block(x, filters=32, kernel_size=3, strides=1, activation='relu', use_dropout=True)
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='same')(x)
        x = self.conv_block(x, filters=64, kernel_size=3, strides=1, activation='relu', use_dropout=True)
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='same')(x)
        x = self.conv_block(x, filters=128, kernel_size=3, strides=1, activation='relu', use_dropout=True)
        x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='same')(x)
        x = self.conv_block(x, filters=256, kernel_size=3, strides=1, activation='relu', use_dropout=True)
        x = GlobalAveragePooling2D()(x)

        self.model = Model(inputs=input_tensor, outputs=x)


    def conv_block(self, x, filters, kernel_size, strides, activation='relu', use_dropout=False):
        #x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if use_dropout:
            x = Dropout(0.5)(x)
        return x


