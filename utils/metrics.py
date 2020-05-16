import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import check_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

from sklearn.metrics import roc_curve, auc

"Function from: https://www.kaggle.com/pednt9/alaska2-srnet-in-keras" \
"and from https://www.kaggle.com/meaninglesslives/alaska2-cnn-multiclass-classifier "

def weighted_auc(y_true, y_valid):
    # Checking if we are using multiclass
    multiclass = np.array(y_valid).shape[1] != 1
    if multiclass:
        y_valid = np.array(y_valid)
        labels = y_valid.argmax(1)
        new_preds = np.zeros((len(y_valid),))

        new_preds[labels != 0] = y_valid[labels != 0, 1:].sum(1)
        new_preds[labels == 0] = 1 - y_valid[labels == 0, 0]
        y_valid = new_preds.tolist()
        y_valid = [[i] for i in y_valid]
        y_true = np.array(y_true)
        y_true[y_true != 0] = 1


    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

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
            if mask.sum() == 0:
                continue
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
    #update_op = streaming_false_positive_rate(y_pred, y_true)
    #print(update_op)
    #return update_op
    return tf.py_function(func=weighted_auc, inp=[y_true, y_pred], Tout=tf.float32)


def multiclass_to_binary(y):
    print("TODO")