import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import os
from utils.metrics import alaska_tf
from model.backbone.UNIWARD import UNIWARD


def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

class RegressionModel:
    def __init__(self, mode, num_classes, input_shape=(512, 512, 3), log_dir='./logs'):
        if mode == "binary":
            self.num_classes = 1
            self.loss = 'mse'
            self.mode = mode
        elif mode == "multiclass":
            self.num_classes = num_classes
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
            self.mode = mode
        else:
            print("Mode not implemented. Exiting")
            exit()
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
        # backbone = UNIWARD(x_input).model
        for layer in backbone.layers:
            layer.trainable = True
        out = Dense(self.num_classes, kernel_initializer='normal')(backbone.output)
        if self.mode == "multiclass":
            out = Activation('softmax')(out)

        model = Model(inputs=backbone.input, outputs=out)
        model_checkpoint = ModelCheckpoint(
            filepath=self.log_dir + '/regression_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
        tb = TensorBoard(log_dir=self.log_dir, update_freq='epoch', profile_batch=100000000)
        self.callbacks = [tb, model_checkpoint]


        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                        verbose=1)
        model.compile(optimizer=adam,
                      loss=self.loss,
                      metrics=[self.loss, alaska_tf])

        self.model = model
        return self.model
