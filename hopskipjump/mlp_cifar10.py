import sys
import pickle
import time
import numpy as np

from sklearn.metrics import accuracy_score
from art.attacks.evasion import HopSkipJump
from art.classifiers import BlackBoxClassifier

import hopskipjump_simclr as hopskipjump
import utils

sys.path.append('..')
from core.scd_lib import *
from tools import args, save_checkpoint, print_title, load_data


class modelWrapper():
    def __init__(self, model):
        self.model = model

    def predict_one_hot(self, x_test):
        pred_y = self.model.predict(x_test)
        pred_one_hot = np.eye(2)[pred_y.astype(int)]

        return pred_one_hot


def loadData(datatype):

    if datatype == 'gtsrb_binary':
        x_train, x_test, y_train, y_test = load_data('gtsrb_binary', 2)
        input_shape = 3*48*48
    elif datatype == 'cifar10_binary':
        x_train, x_test, y_train, y_test = load_data('cifar10_binary', 2)
        input_shape = 3*32*32
    elif datatype == 'cifar10':
        x_train, x_test, y_train, y_test = load_data('cifar10', 2)
        input_shape = 3*32*32

    return x_train, x_test, y_train, y_test, input_shape


def main():

    # Define variable
    datatype = 'cifar10'
    modelpath = '../binary/checkpoints/cifar10_mlp_100.pkl'

    # Define which data sample to be processed
    data_idx = 0
    print('---------------data point---------------\n', data_idx)

    print('init_size=1000')

    # Load data
    x_train, x_test, y_train, y_test, input_shape = loadData(datatype)

    # Load model
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    # Predict
    pred_y = model.predict(x_test)

    print('pred_y: ', pred_y[0], pred_y[1], pred_y[2], pred_y[3], pred_y[4], pred_y[5], pred_y[6])
    print('y_test: ', y_test[0], y_test[1], y_test[2], y_test[3], y_test[4], y_test[5], y_test[6])
    print('\npred_y: ', pred_y[-1], pred_y[-2], pred_y[-3], pred_y[-4], pred_y[-5], pred_y[-6])
    print('y_test: ', y_test[-1], y_test[-2], y_test[-3], y_test[-4], y_test[-5], y_test[-6])
    print('pred_y[{}]: '.format(data_idx), pred_y[data_idx])
    print('y_test[{}]: '.format(data_idx), y_test[data_idx])
    print('Accuracy: ', accuracy_score(y_true=y_test, y_pred=pred_y))


    # Create a model wrapper
    predictWrapper = modelWrapper(model)
    adv_data = hopskipjump.attack(predictWrapper, x_train, x_test, y_train, y_test, input_shape, x_test[data_idx])

    print('adv_data predict: ', model.predict(adv_data))

main()
