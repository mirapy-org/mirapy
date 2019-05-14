from scipy.optimize import minimize
from autograd import grad
from copy import deepcopy


class ParameterEstimation:
    def __init__(self, x, y, model, loss_function, callback=None):
        """
        Base class of parameter estimation of a model using regression.

        :param x: Array of input values.
        :param y: Array of target values.
        :param model: Model instance.
        :param loss_function: Instance of loss function.
        :param callback: Callback function.
        """
        self.x = x
        self.y = y
        self.init_model = deepcopy(model)
        self.model = deepcopy(model)
        self.p_init = model.get_params_as_array()
        self.loss_function = loss_function
        self.callback = callback
        self.results = None

    def regression_function(self, params):
        """
        Return the output of loss function.

        :param params: Array of new parameters of the model.
        :return: Output of loss function.
        """
        self.model.set_params_from_array(params)
        y_true = self.y
        y_pred = self.model(self.x)
        return self.loss_function(y_true, y_pred)

    def get_model(self):
        """
        Returns a copy of model used in estimation.

        :return: Model instance.
        """
        model = deepcopy(self.init_model)
        if self.results is not None:
            model.set_params_from_array(self.results.x)
        return model

    def fit(self):
        """
        Fits the data into the model using regression.

        :return: Returns the result.
        """
        results = minimize(self.regression_function, self.p_init,
                           method='L-BFGS-B',
                           jac=grad(self.regression_function),
                           callback=self.callback)
        self.results = results
        return results
