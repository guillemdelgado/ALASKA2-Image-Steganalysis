import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.metrics import roc_curve, auc

"Function from: https://www.kaggle.com/pednt9/alaska2-srnet-in-keras"

def weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_valid)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    try:
        competition_metric = 0
        for idx, weight in enumerate(weights):
            y_min = tpr_thresholds[idx]
            y_max = tpr_thresholds[idx + 1]
            mask = (y_min < tpr) & (tpr < y_max)

            x_padding = np.linspace(fpr[mask][-1], 1, 100)

            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
            y = y - y_min  # normalize such that curve starts at y=0
            score = auc(x, y)
            submetric = score * weight
            best_subscore = (y_max - y_min) * weight
            competition_metric += submetric
    except:
        # sometimes there's a weird bug so return naive score
        return .5

    return competition_metric / normalization


def alaska_tf(y_true, y_pred):
    """Wrapper for the above function"""
    return tf.py_function(func=weighted_auc, inp=[y_true, y_pred], Tout=tf.float32)