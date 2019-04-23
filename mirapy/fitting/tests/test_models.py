from mirapy.fitting import Gaussian1D
import pytest


def test_gaussian1d_error():
    model = Gaussian1D()

    with pytest.raises(ValueError):
        model.set_params_from_array([])
