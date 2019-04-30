import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam


class Classifier:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def build_model(self, x):
        pass

    def save_model(self, params):
        pass

    def load_model(self):
        pass


class XRayBinaryClassifier(Classifier):
    """
    """
    def __init__(self):
        pass

    def build_model(self, activation='relu', optimizer=Adam(lr=0.0001, decay=1e-6)):
        """
        build model pre-worked
        """
        input_x = Input(shape=(3, 1))
        x = Dense(32, activation = self.activation)(input_x)
        x = Dense(32, activation = self.activation)(x)
        x = Dense(16, activation = self.activation)(x)
        y = Dense(3, activation='softmax')(x)
        self.model = Model(input_x, y)
        self.optimizer = optimizer
        
    def save_model(self, model_name='xrb_model.h5'):
        """
        save model
        """
        path = 'models/' + model_name 
        self.model.save(path)

    def load_model(self, model_name):
        """
        load saved model
        """
        path = 'models/' + model_name 
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")
        
