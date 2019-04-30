import os
import numpy as np
from tqdm import tqdm
import cv2


def load_messier_catalog_images(path, img_size=None, normalize=True):
    # TODO: Allow downloading data from github
    images = []
    for filename in tqdm(os.listdir(path)):
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if normalize:
            img = img/img.max()
            img = img * 255.
        if img_size:
            img = cv2.resize(img, img_size)
        images.append(np.array(img))
    return np.array(images)
