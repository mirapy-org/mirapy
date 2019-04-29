import autograd.numpy as np


class classifier:
    def __init__(self):
        return

    def build_model(self, x):
        return

    def save_model(self, params):
        return

    def load_model(self):
        return


class XrayBinary(classifier):
    """
        One dimensional Gaussian model.
        Parameters
        ----------
        amplitude : float
            Amplitude of the Gaussian.
        mean : float
            Mean of the Gaussian.
        stddev : float
            Standard deviation of the Gaussian.
        """

    def __init__(self, amplitude=1., mean=0., stddev=1.):
        self.amplitude = amplitude
        self.mean = mean
        self.stddev = stddev

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        """
        Gaussian1D model function.
        Parameters
        ----------
        x : array
            Input of the model.
        Returns
        -------
        array : Output of the Gaussian function.
        """
        return self.amplitude * np.exp(-0.5 * (x - self.mean) ** 2 / self.stddev ** 2)

    def set_params_from_array(self, params):
        """
        Sets the parameters of the model from an array.
        """
        if len(params) != 3:
            raise ValueError("The length of the parameter array must be 3")

        self.amplitude = params[0]
        self.mean = params[1]
        self.stddev = params[2]

    def get_params_as_array(self):
        return np.array([self.amplitude, self.mean, self.stddev])

    © 2019 GitHub, Inc.
    Terms
    Privacy
    Security
    Status
    Help

    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

