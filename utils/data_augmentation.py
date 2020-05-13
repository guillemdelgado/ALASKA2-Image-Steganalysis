import numpy as np
import cv2
import random


def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return np.flipud(image_array)


def vertical_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return np.fliplr(image_array)


def rotation(image_array):
    return np.rot90(image_array, random.uniform(1, 3))
