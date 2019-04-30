import numpy as np
import scipy
from keras.preprocessing.image import img_to_array


def get_psf_airy(n, nr):
    xpsf = np.linspace(-1, 1, n)
    xg, yg = np.meshgrid(xpsf, xpsf)
    r = np.sqrt(xg**2+yg**2)*np.pi*nr
    psf = (scipy.special.j1(r)/r)**2
    psf = psf/psf.sum()
    return psf


def image_augmentation(images, image_data_generator, num_of_augumentations):
    images_aug = []
    for image in images:
        img_shape = image.shape
        img_array = img_to_array(image)
        img_array = img_array.reshape((1,) + img_array.shape)
        i = 0
        for batch in image_data_generator.flow(img_array, batch_size=1):
            i += 1
            img = batch[0]
            img = img.reshape(img_shape)
            images_aug.append(img)

            if i >= num_of_augumentations:
                break

    images_aug = np.array(images_aug)
    return images_aug
