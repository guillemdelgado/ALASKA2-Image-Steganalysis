import numpy as np
import pandas as pd

PATH = "D:\\Data\\alaska2-image-steganalysis\\"
path_0 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold0\\fold0_submission48_tta.csv"
path_1 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold1\\fold1_submission48_tta.csv"
path_4 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold4\\fold4_submission45_tta.csv"

sample_sub = pd.read_csv(PATH + 'sample_submission.csv')
sub_0 = pd.read_csv(path_0)
sub_1 = pd.read_csv(path_1)
sub_4 = pd.read_csv(path_4)

sample_sub['Label'] = 0.4 * sub_1['Label'] + 0.4 * sub_4['Label'] + 0.2 * sub_0['Label']
sample_sub.to_csv('fold1-48TTA_fold4-49TTA_fold0-38TTA_submission_blended.csv', index=None)
