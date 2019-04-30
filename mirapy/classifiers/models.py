import os
import warnings
from keras.models import load_model, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
warnings.filterwarnings('ignore')


class Classifier:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.activation = None

    def compile_model(self, activation, optimizer, loss):
        pass

    def save_model(self, model_name, path):
        pass

    def load_model(self, model_name, path):
        pass


class XRayBinaryClassifier(Classifier):
    """
    """
    def compile_model(self, activation='relu',
                      optimizer=Adam(lr=0.0001, decay=1e-6), loss='mean_squared_error'):
        """
        build model pre-worked
        """
        self.activation = activation
        input_x = Input(shape=(3, 1))
        x = Dense(32, activation=self.activation)(input_x)
        x = Dense(32, activation=self.activation)(x)
        x = Dense(16, activation=self.activation)(x)
        y = Dense(3, activation='softmax')(x)
        self.model = Model(input_x, y)
        self.optimizer = optimizer
        self.model.compile(self.optimizer, loss, metrics=['accuracy'])
        
    def save_model(self,  model_name='xrb_model.h5', path='./models'):
        """
        save model
        """
        path = path + '/' + model_name
        self.model.save(path)

    def load_model(self, model_name, path='./models'):
        """
        load saved model
        """
        path = path + '/' + model_name
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")

