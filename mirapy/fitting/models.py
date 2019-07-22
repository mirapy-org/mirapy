import autograd.numpy as np


class Model1D:
    def __init__(self):
        """
        Base class for 1-D model.
        """
        pass

    def __call__(self, x):
        """
        Return the value of evaluate function by calling it.

        :param x: Array of 1-D input values.
        :return: Return the output of the evaluate function.
        """
        return self.evaluate(x)

    def evaluate(self, x):
        """
        Return the value of a model of the given input.

        :param x: Array of 1-D input values.
        :return: Return the output of the model.
        """
        pass

    def set_params_from_array(self, params):
        """
        Sets the parameters of the model from an array.

        :param params: Array of parameter values.
        """
        pass

    def get_params_as_array(self):
        """
        Returns the parameters of the model as an array.
        """
        pass


class Gaussian1D(Model1D):
    def __init__(self, amplitude=1., mean=0., stddev=1.):
        """
        One dimensional Gaussian model.

        :param amplitude: Amplitude.
        :param mean: Mean.
        :param stddev: Standard deviation.
        """
        self.amplitude = amplitude
        self.mean = mean
        self.stddev = stddev

    def evaluate(self, x):
        """
        Return the value of Gaussian model of the given input.

        :param x: Array of 1-D input values.
        :return: Return the output of the model.
        """
        return self.amplitude * np.exp(-0.5 *
                                       (x-self.mean) ** 2 / self.stddev ** 2)

    def set_params_from_array(self, params):
        """
        Sets the parameters of the model from an array.

        :param params: Array of parameter values.
        """
        if len(params) != 3:
            raise ValueError("The length of the parameter array must be 3")

        self.amplitude = params[0]
        self.mean = params[1]
        self.stddev = params[2]

    def get_params_as_array(self):
        """
        Returns the parameters of the model as an array.
        """
        return np.array([self.amplitude, self.mean, self.stddev])
