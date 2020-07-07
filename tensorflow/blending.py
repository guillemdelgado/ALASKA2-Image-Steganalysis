import numpy as np
import pandas as pd

PATH = "D:\\Data\\alaska2-image-steganalysis\\"
path_0 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold0\\fold0_submission48_tta.csv"
path_1 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold1\\fold1_submission52_tta.csv"
path_2 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold2\\fold2_submission50_TTA.csv"
path_3 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold3\\fold3_submission48_TTA.csv"
path_4 = "C:\\Users\\guill\\Documents\\Development\\ALASKA2-Image-Steganalysis\\pytorch\\logs_kfold4\\fold4_submission45_tta.csv"

sample_sub = pd.read_csv(PATH + 'sample_submission.csv')
sub_0 = pd.read_csv(path_0)
sub_1 = pd.read_csv(path_1)
sub_2 = pd.read_csv(path_2)
sub_3 = pd.read_csv(path_3)
sub_4 = pd.read_csv(path_4)

sample_sub['Label'] = 0.35*sub_1['Label']+0.35*sub_4['Label']+0.1*sub_3['Label']+0.1*sub_0['Label']+0.1*sub_2['Label']
sample_sub.to_csv('035fold1-52TTA_035fold3-48TTA_01fold4-49TTA_01fold0-48TTA_01fold2-50TTAsubmission_blended.csv', index=None)
