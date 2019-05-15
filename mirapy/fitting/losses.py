import autograd.numpy as np
from autograd.scipy.stats import norm


def negative_log_likelihood(y_true, y_pred):
    """
    Function for negative log-likelihood error.

    :param y_true: Array of true values.
    :param y_pred: Array of predicted values.
    :return: Float. Loss value.
    """
    ll = norm.logpdf(y_true, y_pred)
    return -np.sum(ll)


def mean_squared_error(y_true, y_pred):
    """
    Function for mean squared error.

    :param y_true: Array of true values.
    :param y_pred: Array of predicted values.
    :return: Float. Loss value.
    """
    return np.sum((y_true-y_pred)**2)/len(y_true)
