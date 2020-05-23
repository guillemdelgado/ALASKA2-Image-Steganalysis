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
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import utils
from data_loader.generator import DataGenerator
from model.regression import RegressionModel
from data_loader.alaska import Alaska

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
utils.seed_everything()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)



PATH = "D:\\Data\\alaska2-image-steganalysis\\"
IMG_SIZE = 512
train_val_ratio = 0.7
batch_size = 4
format = "RGB"
mode = "multiclass"
loading_path = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\multiclass_checkpoints\\regression_epoch-02_loss-0.8505_val_loss-1.0366.h5"

alaska_data = Alaska(PATH, train_val_ratio, mode, multiclass_file='./multiclass_stega_df.csv')
data = alaska_data.build()
IMAGE_IDS_train = data[0]
IMAGE_LABELS_train = data[1]
IMAGE_IDS_val = data[2]
IMAGE_LABELS_val = data[3]

# Test data
test_ids = os.listdir(os.path.join(PATH, 'Test'))
for i in range(len(test_ids)):
    test_ids[i] = os.path.join(os.path.join(PATH, 'Test'), test_ids[i])

sample_sub = pd.read_csv(PATH + 'sample_submission.csv')


train_gen = DataGenerator(IMAGE_IDS_train, IMAGE_LABELS_train, batch_size=batch_size, shuffle=True,
                          sampling="under_sample", data_augmentation=True, format=format)
validation_gen = DataGenerator(IMAGE_IDS_val, IMAGE_LABELS_val, batch_size=batch_size, format=format)
test_gen = DataGenerator(test_ids, None, batch_size=8, format=format)
test_gen_h = DataGenerator(test_ids, None, batch_size=8, format=format, tta='horizontal')
test_gen_v = DataGenerator(test_ids, None, batch_size=8, format=format, tta='vertical')
test_gen_r = DataGenerator(test_ids, None, batch_size=8, format=format, tta='rotate')

print("Loading model")
regression = RegressionModel(mode, alaska_data.num_classes)
model = regression.build_model()
model.summary()
if loading_path != "":
    model.load_weights(loading_path, by_name=True, skip_mismatch=True)

model.fit(x=train_gen,
          steps_per_epoch=len(train_gen),
          validation_data=validation_gen,
          validation_steps=len(validation_gen),
          epochs=10,
          callbacks=regression.callbacks)

output_predictions_o = model.predict(x=test_gen,
                                     steps=len(test_gen))
output_predictions_h = model.predict(x=test_gen_h,
                                     steps=len(test_gen_h))
output_predictions_v = model.predict(x=test_gen_v,
                                     steps=len(test_gen_v))
output_predictions_r = model.predict(x=test_gen_r,
                                     steps=len(test_gen_r))

output = output_predictions_o * 0.4 + output_predictions_h * 0.2 + output_predictions_v * 0.2 + output_predictions_r * 0.2
output_predictions = utils.utils.multiclass_to_binary(output)

if mode == "multiclass":
    output_predictions = utils.utils.multiclass_to_binary(output)
sample_sub['Label'] = output_predictions
sample_sub.to_csv('submission.csv', index=None)


