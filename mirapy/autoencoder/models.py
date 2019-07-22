import os
from keras.models import Model, load_model
from keras.layers import *
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(self):
        """
        Base Class for autoencoder models.
        """
        self.model = None
        self.history = None
        self.dim = None
        self.input_img = None
        self.encoded = None
        self.decoded = None

    def compile(self, optimizer, loss):
        """
        Compile model with given configuration.

        :param optimizer: Instance of optimizer.
        :param loss: String (name of loss function) or custom function.
        """
        pass

    def train(self, x, y, batch_size=32, epochs=100, validation_data=None,
              shuffle=True, verbose=1):
        """
        Trains the model on the training data with given settings.

        :param x:  Numpy array of training data.
        :param y:  Numpy array of target data.
        :param epochs: Integer. Number of epochs during training.
        :param batch_size: Number of samples per gradient update.
        :param validation_data: Numpy array of validation data.
        :param shuffle: Boolean. Shuffles the data before training.
        :param verbose: Value is 0, 1, or 2.
        """
        pass

    def predict(self, x):
        """
        Predicts the output of the model for the given data as input.

        :param x: Input data as Numpy arrays.
        """
        pass

    def plot_history(self):
        """
        Plots loss vs epoch graph.
        """
        plt.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history.keys():
            plt.plot(self.history.history['val_loss'])
        plt.title('Autoencoder loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

    def save_model(self, model_name, path='models/'):
        """
        Saves a model into a H5py file.

        :param model_name: File name.
        :param path: Pa
        """
        path += model_name
        self.model.save(path)

    def load_model(self, model_name, path='models/'):
        """
        Loads a model from a H5py file.

        :param model_name: File name.
        :param path: Pa
        """
        path += model_name
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")

    def summary(self):
        pass


class DeNoisingAutoencoder(Autoencoder):
    def __init__(self, img_dim, activation='relu', padding='same'):
        """
        De-noising Autoencoder used for the astronomical image reconstruction.

        :param img_dim: Set. Dimension of input and output image.
        :param activation: String (activation function name).
        :param padding: String (type of padding in convolution layers).
        """
        self.dim = img_dim
        self.input_img = Input(shape=(*img_dim, 1))

        x = Conv2D(64, (3, 3), activation=activation, padding=padding)(
            self.input_img)
        x = MaxPooling2D((2, 2), padding=padding)(x)
        x = Conv2D(32, (3, 3), activation=activation, padding=padding)(x)
        x = MaxPooling2D((2, 2), padding=padding)(x)
        x = Conv2D(16, (3, 3), activation=activation, padding=padding)(x)
        x = BatchNormalization()(x)
        encoded = MaxPooling2D((2, 2), padding=padding)(x)

        x = Conv2D(16, (3, 3), activation=activation, padding=padding)(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation=activation, padding=padding)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid',
                              padding=padding)(x)

    def compile(self, optimizer, loss):
        """
        Compile model with given configuration.

        :param optimizer: Instance of optimizer.
        :param loss: String (name of loss function) or custom function.
        """
        self.model = Model(self.input_img, self.decoded)
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, x, y, batch_size=32, epochs=100, validation_data=None,
              shuffle=True, verbose=1):
        """
        Trains the model on the training data with given settings.

        :param x:  Numpy array of training data.
        :param y:  Numpy array of target data.
        :param epochs: Integer. Number of epochs during training.
        :param batch_size: Number of samples per gradient update.
        :param validation_data: Numpy array of validation data.
        :param shuffle: Boolean. Shuffles the data before training.
        :param verbose: Value is 0, 1, or 2.
        """
        self.history = self.model.fit(x, y,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      validation_data=validation_data,
                                      verbose=verbose)

    def predict(self, x):
        """
        Predicts the output of the model for the given data as input.

        :param x: Input data as Numpy arrays.
        """
        return self.model.predict(x)

    def show_image_pairs(self, original_images, decoded_images, max_images):
        """
        Displays images in pair of images in grid form using Matplotlib.

        :param original_images: Array of original images.
        :param decoded_images: Array of decoded images.
        :param max_images: Integer. Set number of images in a row.
        """
        n = min(max_images, len(decoded_images))

        plt.figure(figsize=(20, 8))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(original_images[i].reshape(self.dim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_images[i].reshape(self.dim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
