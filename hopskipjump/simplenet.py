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


class modelWrapper():
    def __init__(self, model, datatype):
        self.model = model
        self.datatype = datatype

    def predict_one_hot(self, x_test):

        if self.datatype is 'gtsrb_binary':
            x_test = x_test.reshape(-1, 3, 48, 48)
        elif self.datatype is 'cifar10_binary': # class 6, 8
            x_test = x_test.reshape(-1, 3, 32, 32)

        pred_y = self.model.predict(x_test)
        pred_one_hot = np.eye(2)[pred_y.astype(int)]

        return pred_one_hot


def loadData(datatype):

    if datatype is 'gtsrb_binary':
        x_train, x_test, y_train, y_test = load_data('gtsrb_binary', 2)
        x_train = x_train.reshape((-1, 3, 48, 48))
        x_test = x_test.reshape((-1, 3, 48, 48))
        input_shape = 3*48*48
    elif datatype is 'cifar10_binary':
        x_train, x_test, y_train, y_test = load_data('cifar10_binary', 2)
        x_train = x_train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
        x_test = x_test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
        input_shape = 3*32*32

    return x_train, x_test, y_train, y_test, input_shape


def adv_gen(attack, orginal_data, datatype):
    adv_data = attack.generate(x=orginal_data)

    print('x ', orginal_data)
    print('x_adv ', adv_data)

    dist2 = utils.computeDist2(orginal_data, adv_data)
    print('test data dist2: ', dist2)

    distInf = utils.computeDistInf(orginal_data, adv_data)
    print('test data distInf: ', distInf)


def main():

    # Define variable
    datatype = 'gtsrb_binary'
    modelpath = '../binary/checkpoints/gtsrb_binary_simplenet_100.pkl'

    # Define which data sample to be processed
    data_idx = 0

    # Load data
    x_train, x_test, y_train, y_test, input_shape = loadData(datatype)

    # Load model
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    # Predict
    pred_y = model.predict(x_test)
    print('pred_y: ', pred_y[:10])
    print('y_test: ', y_test[:10])
    print('Accuracy: ', accuracy_score(y_true=y_test, y_pred=pred_y))

    # Create a model wrapper
    predictWrapper = modelWrapper(model, datatype)

    hopskipjump.attack(predictWrapper, x_train, x_test, y_train, y_test, input_shape, x_test[data_idx])


main()
