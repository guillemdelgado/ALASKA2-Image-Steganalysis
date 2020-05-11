import numpy as np
import cv2

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return cv2.flip(image_array, 1)

def vertical_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return cv2.flip(image_array, 0)