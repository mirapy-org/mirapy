from mirapy.fitting.models import Gaussian1D
import pytest


def test_Gaussian1D_error():
    model = Gaussian1D()

    with pytest.raises(ValueError):
        model.set_params_from_array([])
