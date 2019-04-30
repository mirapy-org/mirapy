import os
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
import cv2


def load_messier_catalog_images(path, img_size=None, disable_tqdm=False):
    # TODO: Allow downloading data from github repo
    images = []
    for filename in tqdm(os.listdir(path), disable=disable_tqdm):
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = img/img.max()
        img = img * 255.
        if img_size:
            img = cv2.resize(img, img_size)
        images.append(np.array(img))
    return np.array(images)


def prepare_messier_catalog_images(images, psf, sigma, normalize=True):
    if normalize:
        images = np.array(images).astype('float32') / 255.

    x_conv2d = [convolve2d(I, psf, 'same') for I in tqdm(images)]
    x_conv2d_noisy = [I + sigma * np.random.poisson(I) for I in tqdm(x_conv2d)]
    return np.array(x_conv2d_noisy)
