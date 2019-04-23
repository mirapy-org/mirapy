from scipy.optimize import minimize
from autograd import grad
from copy import deepcopy


class ParameterEstimation():
    def __init__(self, x, y, model, loss_function, callback=None):
        self.x = x
        self.y = y
        self.model = deepcopy(model)
        self.p_init = model.get_params_as_array()
        self.loss_function = loss_function
        self.callback = callback

    def regression_function(self, params):
        self.model.set_params_from_array(params)
        y_true = self.y
        y_pred = self.model(self.x)
        return self.loss_function(y_true, y_pred)

    def fit(self):
        results = minimize(self.regression_function, self.p_init, method='L-BFGS-B',
                           jac=grad(self.regression_function), callback=self.callback)
        return results
