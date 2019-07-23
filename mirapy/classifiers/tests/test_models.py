import os
import pytest
import numpy as np
from mirapy.data import load_dataset
from mirapy.classifiers import models

def test_XRayBinaryClassifier():
    x = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    y = np.array([0, 1, 2])
    model = models.XRayBinaryClassifier('relu')
    model.compile()
    model.train(x, y)
    y_pred = model.predict(x)
    
    assert type(y_pred) == np.ndarray