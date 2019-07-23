from mirapy.fitting import Model1D, Gaussian1D
import pytest


def test_model1d():
    model = Model1D()
    model.evaluate([])
    model.get_params_as_array()
    model.set_params_from_array([])

def test_gaussian1d_error():
    model = Gaussian1D()

    with pytest.raises(ValueError):
        model.set_params_from_array([])
