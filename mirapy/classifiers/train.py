import os
import argparse
import numpy as np
from keras.optimizers import Adam
from models import *
from utils import load_data

parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_directory', type=str, required=True,
                    help='data-set directory')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store models weights')
parser.add_argument('--model_name', type=str, default='xrb_model.h5',
                    help='saved model name')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--activation', type=str, default='relu',
                    help='activation function for hidden layers')
parser.add_argument('--optimizer', type=str, default=Adam(lr=0.0001, decay=1e-6),
                    help='activation function for hidden layers')
parser.add_argument('--validation_split', type=float, default=0.1,
                    help='validation-data split')
parser.add_argument('--test_split', type=float, default=0.2,
                    help='test-data split(float)')

args = parser.parse_args()

if not os.path.isdir(args.data_directory):
    raise OSError('Directory '+str(args.data_directory)+' does not exist.')

if not os.path.isdir(args.save_dir):
    raise OSError('Directory '+str(args.save_dir)+' does not exist.')

if args.test_split >= 1 or args.test_split <= 0:
    raise ValueError('test_split value should be a fraction')


def train(model, x_train, y_train):
    if not type(x_train) == np.ndarray and type(x_test) == np.ndarray and \
                    type(y_train) == np.ndarray and type(y_test) == np.ndarray:
        raise ValueError('Input array should be numpy arrays')
    history = model.fit(x_train, y_train, epochs=args.num_epochs, shuffle=True,
                        batch_size=args.batch_size, validation_split=args.validation_split)


if __name__ == '__main__':
    x_train, x_test, _, _ = load_data(args.data_directory, args.test_split)
    xrb = XRayBinaryClassifier()
    xrb.compile_model(args.activation, args.optimizer)
    train(xrb.model, x_train, x_test)
    xrb.save_model(args.save_dir, args.model_name)
