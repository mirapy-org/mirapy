import os
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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


def prepare_messier_catalog_images(images, psf, sigma):
    images = np.array(images).astype('float32') / 255.
    x_conv2d = [convolve2d(I, psf, 'same') for I in images]
    x_conv2d_noisy = [I + sigma * np.random.poisson(I) for I in x_conv2d]
    return images, x_conv2d_noisy


def load_xray_binary_data(path, test_split, standard_scaler=True):
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
            datapoints[i][0] = '0'
        elif system in pulsar_keys:
            datapoints[i][0] = '1'
        elif system in nonpulsar_keys:
            datapoints[i][0] = '2'

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

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=test_split, random_state=42)

    y_train = to_categorical(y_train)
    return x_train, y_train, x_test, y_test


def load_atlas_star_data(path, test_split, standard_scaler=True,
                         feat_list=None):
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

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=test_split, random_state=42)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y_train = onehot_encoded

    return x_train, y_train, x_test, y_test
