import sys
import pickle
import time
import numpy as np

from sklearn.metrics import accuracy_score
from art.attacks.evasion import HopSkipJump
from art.classifiers import BlackBoxClassifier

import hopskipjump
import utils

sys.path.append('..')
from core.scd_lib import *
from tools import args, save_checkpoint, print_title, load_data


class modelWrapper():
    def __init__(self, model, datatype, vote):
        self.model = model
        self.datatype = datatype
        self.vote = vote

    def predict_one_hot(self, x_test):
        if self.datatype == 'gtsrb_binary':
            x_test = x_test.reshape(-1, 3, 48, 48)
        elif self.datatype == 'cifar10':
            x_test = x_test.reshape(-1, 3, 32, 32)

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
        x_train = x_train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
        x_test = x_test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
        input_shape = 3*32*32

    return x_train, x_test, y_train, y_test, input_shape


def main():

    # Define variable
    datatype = 'cifar10'
    modelpath = '../binary/checkpoints/cifar10_resnet50_10.pkl'

    print('------------- model -------------\n', modelpath)

    # Define which data sample to be processed
    data_idx = 0
    print('---------------data point---------------\n', data_idx)

    # Load data
    x_train, x_test, y_train, y_test, input_shape = loadData(datatype)

    # Load model
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    adv_lst = []

    # Predict
    for vote in range(100):
        print('\n\nVote id: {}\n'.format(vote))
        pred_y = model.predict(x_test, best_index=vote).astype(int)

        print('pred_y[{}]: '.format(data_idx), pred_y[data_idx])
        print('y_test[{}]: '.format(data_idx), y_test[data_idx])
        print('Accuracy: ', accuracy_score(y_true=y_test, y_pred=pred_y))

        # Create a model wrapper
        predictWrapper = modelWrapper(model, datatype, vote)
        adv_data = hopskipjump.attack(predictWrapper, x_train, x_test, y_train, y_test, input_shape, x_test[data_idx])

        adv_lst.append(adv_data)

        if datatype == 'gtsrb_binary':
            adv_data = adv_data.reshape(-1, 3, 48, 48)
        elif datatype == 'cifar10':
            adv_data = adv_data.reshape(-1, 3, 32, 32)
        print('adv_data predict: ', model.predict(adv_data, best_index=vote))

    adv = np.array(adv_lst)
    print('shape', adv.shape)
    adv = np.squeeze(adv, axis=1)
    print('shape', adv.shape)
    np.save('resnet_adv_data', adv)


main()
