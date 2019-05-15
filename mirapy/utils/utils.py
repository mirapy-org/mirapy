import numpy as np
import scipy
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array


def get_psf_airy(n, nr):
    """
    Calculates Point Spread Function.

    :param n:
    :param nr:
    :return: Numpy array of Point Spread Function
    """
    xpsf = np.linspace(-1, 1, n)
    xg, yg = np.meshgrid(xpsf, xpsf)
    r = np.sqrt(xg**2+yg**2)*np.pi*nr
    psf = (scipy.special.j1(r)/r)**2
    psf = psf/psf.sum()
    return psf


def image_augmentation(images, image_data_generator, num_of_augumentations,
                       disable=False):
    """
    Form augmented images for input array of images

    :param images: numpy array of Images.
    :param image_data_generator: Keras image generator object.
    :param num_of_augumentations: Number of augmentations of each image.
    :param disable: Bool. Disable/enable tqdm progress bar.
    :return: Numpy array of augmented images.
    """
    images_aug = []
    for image in tqdm(images, disable=disable):
        img_dim = image.shape
        img_array = img_to_array(image)
        img_array = img_array.reshape((1,) + img_array.shape)
        i = 0
        for batch in image_data_generator.flow(img_array, batch_size=1):
            i += 1
            img = batch[0]
            img = img.reshape(img_dim)
            images_aug.append(img)

            if i >= num_of_augumentations:
                break

    images_aug = np.array(images_aug)
    return images_aug


def psnr(img1, img2):
    """
    Calculate Peak Signal to Noise Ratio value.

    :param img1: Float. Array of first image.
    :param img2: Float.Array of second image.
    :return: Float. PSNR value of x and y.
    """
    mse = np.mean((img1 - img2) ** 2)
    return -10 * np.log10(mse)


def append_one_to_shape(x):
    """
    Reshapes input.

    :param x: Array input.
    :return: Reshaped array.
    """
    x_shape = x.shape
    x = x.reshape((len(x), np.prod(x.shape[1:])))
    x = np.reshape(x, (*x_shape, 1))
    return x


def unpickle(file):
    """
    Unpickle and read file.

    :param file: Pickle file to read.
    :return: Data loaded from pickle file.
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def to_numeric(y):
    """
    Convert numpy array of array of probabilities to numeric array.

    :param y: Numpy array.
    :return: Numpy array of classes.
    """
    return np.array([np.argmax(value) for value in y])


def accuracy_per_class(y_true, y_pred):
    """
    Computes accuracy per class.

    :param y_true: True class.
    :param y_pred: Predicted class.
    :return:
    """
    y_true = to_numeric(y_true)
    y_pred = to_numeric(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    return cm.diagonal() / cm.sum(axis=1)
