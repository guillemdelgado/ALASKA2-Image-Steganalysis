import numpy as np
import tensorflow as tf
import os
import jpegio as jpio

# To make things reproductible


def seed_everything(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_KERAS'] = '1'


def JPEGdecompressYCbCr(jpegStruct):
    """
    Reads DCT coefficients and converts them in YCrCb. Source:
    https://www.kaggle.com/remicogranne/jpeg-explanations-ycbcr-qualityfactor-meaning

    :param jpegStruct: jpegio object
    :return: YCrCb Image
    """
    nb_colors = len(jpegStruct.coef_arrays)

    [Col, Row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * Col + 1) * Row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    sz = np.array(jpegStruct.coef_arrays[0].shape)

    imDecompressYCbCr = np.zeros([sz[0], sz[1], nb_colors])
    szDct = (sz / 8).astype('int')

    for ColorChannel in range(nb_colors):
        tmpPixels = np.zeros(sz)

        DCTcoefs = jpegStruct.coef_arrays[ColorChannel]
        if ColorChannel == 0:
            QM = jpegStruct.quant_tables[ColorChannel]
        else:
            QM = jpegStruct.quant_tables[1]

        for idxRow in range(szDct[0]):
            for idxCol in range(szDct[1]):
                D = DCTcoefs[idxRow * 8:(idxRow + 1) * 8, idxCol * 8:(idxCol + 1) * 8]
                tmpPixels[idxRow * 8:(idxRow + 1) * 8, idxCol * 8:(idxCol + 1) * 8] = np.dot(np.transpose(T),
                                                                                             np.dot(QM * D, T))
        imDecompressYCbCr[:, :, ColorChannel] = tmpPixels
    return imDecompressYCbCr
