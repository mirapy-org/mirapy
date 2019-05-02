<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b77eef0121943937dafd94c4b88a3f23a972c5b1
import numpy as np
import scipy
from tqdm import tqdm
from keras.preprocessing.image import img_to_array
<<<<<<< HEAD
=======
# Licensed under the MIT license - see LICENSE.rst
>>>>>>> 5c028025dbaa4d0a605c335f22346b3dd77ac535
=======
>>>>>>> b77eef0121943937dafd94c4b88a3f23a972c5b1


def get_psf_airy(n, nr):
    xpsf = np.linspace(-1, 1, n)
    xg, yg = np.meshgrid(xpsf, xpsf)
    r = np.sqrt(xg**2+yg**2)*np.pi*nr
    psf = (scipy.special.j1(r)/r)**2
    psf = psf/psf.sum()
    return psf


def image_augmentation(images, image_data_generator, num_of_augumentations,
                       disable=False):
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


def psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return -10 * np.log10(mse)


def append_one_to_shape(x):
    x_shape = x.shape
    x = x.reshape((len(x), np.prod(x.shape[1:])))
    x = np.reshape(x, (*x_shape, 1))
    return x
