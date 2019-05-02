from mirapy.fitting import Gaussian1D, mean_squared_error, ParameterEstimation
import autograd.numpy as np
from astropy.modeling import models, fitting


def test_parameter_estimation():
    x = np.linspace(-10., 10., 200)

    amplitude = 3.
    x_0 = 4.
    sigma = 2.
    noise = 0.2

    model = Gaussian1D(amplitude, x_0, sigma)
    y = model(x)

    np.random.seed(0)
    y += np.random.normal(0., noise, x.shape)

    # parameter estimation using MiraPy
    init_model = Gaussian1D(1., 1., 1.)
    parest = ParameterEstimation(x, y, init_model, mean_squared_error)
    parest.fit()
    best_model = parest.get_model()

    # paramter estimation using Astropy

    g_init = models.Gaussian1D(amplitude=1., mean=1., stddev=1.)
    pfit = fitting.LevMarLSQFitter()
    new_model = pfit(g_init, x, y)

    assert np.all(np.isclose(best_model(x), new_model(x), atol=0.01))
