import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import os


class RegressionModel:
    def __init__(self, input_shape=(512,512,3), log_dir='./logs'):
        self.input_shape = input_shape
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model = None

    def build_model(self):
        x_input = Input(shape=self.input_shape, name='input_1')
        resnet = tf.keras.applications.ResNet101V2(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=x_input,
                                                   input_shape=None,
                                                   pooling='avg',
                                                   classes=1)
        out = Dense(1, kernel_initializer='normal')(resnet.output)
        model = Model(inputs=resnet.input, outputs=out)

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse'])

        self.model = model
        return self.model
