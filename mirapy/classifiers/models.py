import os
import numpy as np
from keras.optimizers import *
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense
import warnings
warnings.filterwarnings('ignore')


class Classifier:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.activation = None

    def build_model(self, x):
        pass

    def save_model(self, model_name, path):
        pass

    def load_model(self, model_name, path):
        pass

    def train(self, x_train, y_train, epochs=100, batch_size=32,
              validation_split=0.1):
        pass


class XRayBinaryClassifier(Classifier):
    """
    build pre-worked model
    """
    def __init__(self, activation='relu',
                 optimizer=Adam(lr=0.0001, decay=1e-6)):
        self.activation = activation
        self.optimizer = optimizer

        model = Sequential()
        model.add(Dense(32, input_shape=(3,), activation=self.activation))
        model.add(Dense(32, activation=self.activation))
        model.add(Dense(16, activation=self.activation))
        model.add(Dense(3, activation='softmax'))
        self.model = model

    def compile(self, loss='mean_squared_error'):
        """
        build the model
        """
        self.model.compile(self.optimizer,
                           loss=loss, metrics=['accuracy'])

    def save_model(self, model_name, path='models/'):
        """
        save model
        """
        path = 'models/' + model_name
        self.model.save(path)

    def load_model(self, model_name, path='models/'):
        """
        load saved model
        """
        path = 'models/' + model_name
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")

    def train(self, x_train, y_train, epochs=100, batch_size=32,
              validation_split=0.1):

        if not isinstance(x_train, np.ndarray) and\
                isinstance(y_train, np.ndarray):
            raise ValueError('Input array should be numpy arrays')

        self.model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                       batch_size=batch_size,
                       validation_split=validation_split)

    def test(self, x_test):
        return self.model.predict_classes(x_test)