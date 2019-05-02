import os
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

    def save_model(self, path, model_name):
        pass

    def load_model(self, path, model_name):
        pass


class XRayBinaryClassifier(Classifier):
    """
    """
    def __init__(self):
        pass

    def compile(self, activation, optimizer):
        """
        build model pre-worked
        """
        self.activation = activation
        input_x = Input(shape=(3, ))
        x = Dense(32, activation=self.activation)(input_x)
        x = Dense(32, activation=self.activation)(x)
        x = Dense(16, activation=self.activation)(x)
        y = Dense(3, activation='softmax')(x)
        self.model = Model(input_x, y)
        self.optimizer = optimizer
        self.model.compile(self.optimizer,
                           loss='mean_squared_error', metrics=['accuracy'])

    def save_model(self, path, model_name):
        """
        save model
        """
        path = 'models/' + model_name
        self.model.save(path)

    def load_model(self, path, model_name):
        """
        load saved model
        """
        path = 'models/' + model_name
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")
