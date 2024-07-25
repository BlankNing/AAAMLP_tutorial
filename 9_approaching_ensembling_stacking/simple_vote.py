import numpy as np

def mean_predictions(probas):
    """
    Create mean predictions.

    :param probas: 2-d array of probability values.
    :return: mean probability.
    """
    return np.mean(probas, axis=1)


def max_voting(preds):
    """
    Create max voted predictions.

    :param preds: 2-d array of prediction values.
    :return: max voted predictions.
    """
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)
