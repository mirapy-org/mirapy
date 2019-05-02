import os
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model, Model
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
        input_x = Input(shape=(3,))
        x = Dense(32, activation=self.activation)(input_x)
        x = Dense(32, activation=self.activation)(x)
        x = Dense(16, activation=self.activation)(x)
        y = Dense(3, activation='softmax')(x)
        self.model = Model(input_x, y)

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

        if not isinstance(x_train) == np.ndarray and\
                isinstance(y_train) == np.ndarray:
            raise ValueError('Input array should be numpy arrays')

        self.model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                       batch_size=batch_size,
                       validation_split=validation_split)
