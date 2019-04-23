import autograd.numpy as np
from autograd.scipy.stats import norm


def negative_log_likelihood(y_true, y_pred):
    ll = norm.logpdf(y_true, y_pred)
    return -np.sum(ll)


def mean_squared_error(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)/len(y_true)
