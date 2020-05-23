import numpy as np
import pandas as pd

PATH = "D:\\Data\\alaska2-image-steganalysis\\"
path_1 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\\multiclass_checkpoints\\submission_epoch2_val103.csv"
path_2 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\resnet_checkpoint\oversampled_resnet_withdataugment\\submission_epoch05.csv"

sample_sub = pd.read_csv(PATH + 'sample_submission.csv')
sub_1 = pd.read_csv(path_1)
sub_2 = pd.read_csv(path_2)

sample_sub['Label'] = 0.5 * sub_1['Label'] + 0.5 * sub_2['Label']
sample_sub.to_csv('submission_blended.csv', index=None)
