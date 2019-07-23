import os
import pytest
import numpy as np
import cv2
from mirapy.utils import get_psf_airy
from mirapy.data import load_dataset


def test_load_xray_binary_data():
    path = 'test_XRayBinary/'
    filename = 'test.asc'
    
    os.mkdir(path)
    with open(path+filename, 'w') as f:
        f.write('J0352+309 Pulsar      50139.6     0.363137     0.995448      1.54628\n') 
    x, y = load_dataset.load_xray_binary_data(path)
    
    os.remove(path+filename)
    os.rmdir(path)

    assert len(x) == 1 and len(x) == len(y)

def test_messier_catalog_images():
    path = 'test_messier_catalog_image/'
    filename = 'test.png'
    img = np.zeros([100,100,3],dtype=np.uint8)
    
    os.mkdir(path)
    cv2.imwrite(path+filename, img)

    imgs = load_dataset.load_messier_catalog_images(path)

    os.remove(path+filename)
    os.rmdir(path)

    assert len(imgs) == 1

    psf = get_psf_airy(100, 2)
    imgs, imgs_noisy = load_dataset.prepare_messier_catalog_images(imgs, psf, psf)

    assert len(imgs) == len(imgs_noisy)
