import cv2
from torch.utils.data import Dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import itertools
import pandas as pd

class Alaska2Dataset(Dataset):

    def __init__(self, data, labels, augmentations=None, sampling=None, test=False):

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

        if test:
            self.data = pd.DataFrame({'ImageFileName': list(self.data)}, columns=['ImageFileName'])
        else:
            self.data = pd.DataFrame(list(zip(self.data, self.labels)), columns=['ImageFileName', 'Label'])
        self.augment = augmentations
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.test:
            fn = self.data.loc[idx][0]
        else:
            fn, label = self.data.loc[idx]
        im = cv2.imread(fn)[:, :, ::-1]
        if self.augment:
            # Apply transformations
            im = self.augment(image=im)
        if self.test:
            return im
        else:
            return im, label