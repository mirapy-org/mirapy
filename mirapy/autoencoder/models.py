from keras.models import Model
from keras.layers import *
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(self):
        self.model = None
        self.history = None
        self.dim = None

    def compile(self):
        pass

    def train(self, input_train, input_test, batch_size, epochs):
        pass

    def predict(self):
        return

    def evaluate(self):
        return

    def plot_history(self):
        plt.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history.keys():
            plt.plot(self.history.history['val_loss'])
        plt.title('Autoencoder loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

    def load_model(self, path, model_name):
        pass

    def save_model(self, path, model_name):
        pass

    def summary(self):
        pass


class DeNoisingAutoencoder(Autoencoder):
    def __init__(self, img_dim, activation='relu', padding='same'):
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
        self.model = Model(self.input_img, self.decoded)
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, x, y, batch_size=32, epochs=100, validation_data=None,
              shuffle=True, verbose=1):
        self.history = self.model.fit(x, y,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      validation_data=validation_data,
                                      verbose=verbose)

    def predict(self, x):
        return self.model.predict(x)

    def show_image_pairs(self, original_images, decoded_images, max_images):
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
