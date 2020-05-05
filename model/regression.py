import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import os
from utils.metrics import alaska_tf

class RegressionModel:
    def __init__(self, input_shape=(512, 512, 3), log_dir='./logs'):
        self.input_shape = input_shape
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model = None
        self.callbacks = None

    def build_model(self):
        x_input = Input(shape=self.input_shape, name='input_1')
        backbone = tf.keras.applications.DenseNet121(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=x_input,
                                                   input_shape=None,
                                                   pooling='avg',
                                                   classes=1)
        for layer in backbone.layers:
            layer.trainable = False
        out = Dense(1, kernel_initializer='normal')(backbone.output)
        model = Model(inputs=backbone.input, outputs=out)
        model_checkpoint = ModelCheckpoint(
            filepath=self.log_dir + '/regression_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
        tb = TensorBoard(log_dir=self.log_dir, update_freq='epoch', profile_batch = 100000000)
        self.callbacks = [tb, model_checkpoint]
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse', alaska_tf])

        self.model = model
        return self.model
