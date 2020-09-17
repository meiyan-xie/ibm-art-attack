import sys
import pickle
import time
import numpy as np

from sklearn.metrics import accuracy_score
from art.attacks.evasion import HopSkipJump
from art.classifiers import BlackBoxClassifier

import hopskipjump
import utils as utils

sys.path.append('..')
from core.scd_lib import *
from tools import args, save_checkpoint, print_title, load_data
from core.bnn import *


class modelWrapper():
    def __init__(self, model, vote):
        self.model = model
        self.vote = vote

    def predict_one_hot(self, x_test):
        pred_y = self.model.predict(x_test, best_index=self.vote)
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
    model = BNN(['../binary/checkpoints/cifar10_mlpbnn_approx_%d.h5' % (i) for i in range(100)])

    print('------------- model -------------\n', 'cifar10_mlpbnn_approx')

    # Define which data sample to be processed
    data_idx = 0
    print('---------------data point---------------\n', data_idx)

    # Load data
    x_train, x_test, y_train, y_test, input_shape = loadData(datatype)


    # Predict
    vote = {}
    print('\n\nVote id: {}\n')
    pred_y = model.predict(x_test, best_index=vote).astype(int)

    print('pred_y[0]: ', pred_y[data_idx])
    print('y_test[0]: ', y_test[data_idx])
    print('Accuracy: ', accuracy_score(y_true=y_test, y_pred=pred_y))

    # Create a model wrapper
    predictWrapper = modelWrapper(model, vote)
    adv_data = hopskipjump.attack(predictWrapper, x_train, x_test, y_train, y_test, input_shape, x_test[data_idx])

    print('adv_data predict: ', model.predict(adv_data, best_index=vote))

    np.save('bnn_adv_data/bnn_adv_data_{}', adv_data)


main()
