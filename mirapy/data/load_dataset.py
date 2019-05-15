import os
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mirapy.utils import unpickle


def load_messier_catalog_images(path, img_size=None, disable_tqdm=False):
    """
    Data loader for Messier catalog images. The images are available
    in `messier-catalog-images` repository of MiraPy organisation.

    :param path: String. Directory path.
    :param img_size: Final dimensions of the image.
    :param disable_tqdm: Boolean. Set True to disable progress bar.
    :return: Array of images.
    """
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


def prepare_messier_catalog_images(images, psf, sigma):
    """
    Function to apply convolution and add noise from poisson distribution on
    an array of images.

    :param images: Array of images.
    :param psf: Point Spread Function (PSF).
    :param sigma: Float. VStandard deviation.
    :return: Original image arrays and convolved image arrays.
    """
    images = np.array(images).astype('float32') / 255.
    x_conv2d = [convolve2d(I, psf, 'same') for I in images]
    x_conv2d_noisy = [I + sigma * np.random.poisson(I) for I in x_conv2d]
    return images, x_conv2d_noisy


def load_xray_binary_data(path, standard_scaler=True):
    """
    Loads X Ray Binary dataset from directory.

    :param path: Path to the directory.
    :param standard_scaler: Bool. Standardize data or not.
    :return: Dataset and Class labels.
    """
    asc_files = [os.path.join(dp, f)
                 for dp, dn, filenames in os.walk(path)
                 for f in filenames if os.path.splitext(f)[1] == '.asc']
    datapoints = []
    for path in asc_files:
        with open(path, 'r') as f:
            lis = [line.split() for line in f]
            for l in lis:
                if len(l) == 6:
                    l[1] = l[0] + " " + l[1]
                    l.remove(l[0])
            datapoints += lis

    bh_keys = ['CygX-1 HMBH', 'LMCX-1 HMBH', 'J1118+480 LMBH',
               'J1550m564 LMBH', 'J1650-500 LMBH', 'J1655-40 LMBH',
               'GX339-4 LMBH', 'J1859+226 LMBH', 'GRS1915+105 LMBH']
    pulsar_keys = ['J0352+309 Pulsar', 'J1901+03 Pulsar', 'J1947+300 Pulsar',
                   'J2030p375 Pulsar', 'J1538-522 Pulsar', 'CenX-3 Pulsar',
                   'HerX-1 Pulsar', 'SMCX-1 Pulsar', 'VelaX-1 Pulsar']
    nonpulsar_keys = ['ScoX-1 Zsource', 'GX9+1 Atoll', 'GX17+2 Zsource',
                      'CygX-2 Zsource', 'GX9+9 Atoll', 'GX349+2 Zsource']

    for i, _ in enumerate(datapoints):
        system = datapoints[i][0]
        if system in bh_keys:
            datapoints[i][0] = 'BH'
        elif system in pulsar_keys:
            datapoints[i][0] = 'P'
        elif system in nonpulsar_keys:
            datapoints[i][0] = 'NP'

    rawdf = pd.DataFrame(datapoints)
    rawdf.columns = ['class', 'date', 'intensity', 'c1', 'c2']
    rawdf = rawdf.drop('date', 1)
    rawdf = rawdf.convert_objects(convert_numeric=True)
    df = rawdf.copy()

    scale_features = ['intensity', 'c1', 'c2']
    if standard_scaler:
        ss = StandardScaler()
        df[scale_features] = ss.fit_transform(df[scale_features])

    x = df.drop('class', axis=1).values
    y = df['class'].values

    return x, y


def load_atlas_star_data(path, standard_scaler=True, feat_list=None):
    """
    Loads ATLAS variable star dataset from directory.

    :param path: Path to the directory.
    :param standard_scaler: Bool. Standardize data or not.
    :param feat_list: List of features to include in dataset.
    :return: Dataset and Class labels.
    """
    df = pd.read_csv(path)
    y = df['CLASS']

    # features selected using GradientBoost feature selection
    # (non-zero second decimal place)

    if feat_list is None:
        feat_list = ["fp_timerev", "fp_powerterm", "fp_phase180",
                     "fp_hifreq", "fp_PPFAPshort1", "fp_period",
                     "fp_fournum", "fp_multfac", "vf_percentile10",
                     "fp_PPFAPshort3", "fp_PPFAPshort4", "vf_S_K",
                     "ls_Cchin", "vf_wsd", "vf_percentile75",
                     "fp_domperiod", "ls_RMS", "ls_Pday", "vf_percentile25",
                     "fp_magrms_o", "fp_origLogFAP", "vf_percentile5"]

    list_cols = list(df)
    for f in feat_list:
        if f not in list_cols:
            raise AssertionError("Key "+f + " not in dataframe")

    for f in list_cols:
        if f in feat_list:
            continue
        df.drop(f, axis=1, inplace=True)

    x = df.iloc[:, 0:]
    y = y.values
    x = x.values

    if standard_scaler:
        sc = StandardScaler()
        x = sc.fit_transform(x)

    return x, y


# handle class inequality
def load_ogle_dataset(path, classes, time_len=50, pad=False):
    """
    Loads OGLE variable star time series data from directory.

    :param path: Path to the directory.
    :param classes: Classes to include in dataset.
    :param time_len: Length of time series data.
    :param pad: Bool. Pad zeroes or not.
    :return: Dataset and Class labels.
    """
    mag, y = [], []
    for class_ in classes:
        folder = path + '/' + class_ + '/I'
        for file in os.listdir(folder):
            num_lines = sum(1 for line in open(folder + '/' + file))
            mag_i, j = [0 for i in range(time_len)], 0

            if not pad and num_lines < time_len:
                continue
            for line in open(folder + '/' + file):
                try:
                    _, b, _ = line.split(' ')
                except Exception:
                    break
                mag_i[j] = float(b)
                j += 1
                if j is time_len or j is num_lines:
                    mag.append(np.array(mag_i))
                    y.append(class_)
                    break

    mag = np.array(mag)
    mag = mag.reshape(mag.shape[0], mag.shape[1], 1)
    return mag, y


def load_htru1_data(data_dir='htru1-batches-py'):
    x_train = None
    y_train = []

    for i in range(1, 6):
        x_train_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            x_train = x_train_dict[b'data']
        else:
            x_train = np.vstack((x_train, x_train_dict[b'data']))
        y_train += x_train_dict[b'labels']

    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_train = np.rollaxis(x_train, 1, 4)
    y_train = np.array(y_train)

    x_test_dict = unpickle(data_dir + "/test_batch")
    x_test = x_test_dict[b'data']
    y_test = x_test_dict[b'labels']

    x_test = np.array(x_test).reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test
