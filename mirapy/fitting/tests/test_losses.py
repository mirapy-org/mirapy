from mirapy.fitting.losses import *
import pytest


def test_losses():
    a = np.random.rand(10)
    b = np.random.rand(10)
    
    loss1 = negative_log_likelihood(a, b)
    loss2 = mean_squared_error(a, b)
    
    assert (loss1 > 0.0)
    assert (loss2 > 0.0)