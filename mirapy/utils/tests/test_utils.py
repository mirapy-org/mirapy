import os
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
from mirapy import utils
import pytest


def test_get_psf_airy():
    a = utils.get_psf_airy(1, 1)
    b = np.array([[1.]])
    assert (a == b).all()

def test_psnr():
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    psnr = utils.psnr(a, b)
    assert psnr >= 0 and psnr <= 100

def test_image_augmentation():
    a = np.array([np.random.rand(128, 128)])
    datagen = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
    b = utils.image_augmentation(a, datagen, 10)
    print(type(b))
    assert type(a) == type(b)

def test_append_one_to_shape():
    a = np.array([[1, 2], [3, 4]])
    b = utils.append_one_to_shape(a)
    assert ((a.shape + (1,)) == b.shape)

def test_unpickle():
    a = np.array([1, 2, 3, 4, 5])
    filename = "test.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(a, f)

    b = utils.unpickle(filename)
    os.remove(filename)
    
    assert (a == b).all()

def test_to_numeric():
    a = np.array([[0.2, 0.8], [0.6, 0.4]])
    b = np.array([1, 0])
    assert (utils.to_numeric(a) == b).all()

def test_accuracy_per_class():
    a = utils.accuracy_per_class(np.array([1, 2]), np.array([3, 4]))
    b = np.array([1.])
    assert (a == b).all()
