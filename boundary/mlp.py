import logging
logging.getLogger().setLevel(logging.DEBUG)

import sys
import pickle
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from art.attacks.evasion import BoundaryAttack, FastGradientMethod
from art.classifiers import SklearnClassifier, BlackBoxClassifier
from art.utils import load_mnist
# import utils3 as utils
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


# Boundary attack
def attack():
    x_train, x_test, y_train, y_test = load_data('cifar10', 2)
    # x_train, y_train, x_test, y_test = utils.loadData()
    min_pixel_value = x_train.min()
    max_pixel_value = x_train.max()
    print('min_pixel_value ', min_pixel_value)
    print('max_pixel_value ', max_pixel_value)

    s = time.time()

    path = '../binary/checkpoints/cifar10_scd01mlp_100_br02_nr075_ni1000_i1_ep2.pkl'
    # path = '/home/y/yx277/research/scd01mc/binary/checkpoints/cifar10_mlp.pkl'
    # path = '/research/datasci/mx42/adversarial_machine_learning/IBM_ART/checkpoints_scd_01/scd_stl10_01_v7.pkl'
    with open(path, 'rb') as f:
        model = pickle.load(f)


    pred_y = model.predict(x_test)
    # np.savetxt('pred_y_mpl2', pred_y)
    print('pred_y: ', pred_y)

    # Create a model wrapper
    predictWrapper = modelWrapper(model)

    classifier = BlackBoxClassifier(predict=predictWrapper.predict_one_hot,
                                    input_shape=(32*32*3, ),
                                    nb_classes=2,
                                    clip_values=(min_pixel_value, max_pixel_value))


    print('----- generate adv data -----')
    attack = BoundaryAttack(estimator=classifier, targeted=False, delta=0.01, epsilon=0.01, max_iter=500, num_trial=100, sample_size=100, init_size=100)


    print('----- generate adv test data -----')
    x_test = x_test[0]
    # Input data shape should be 2D
    x_test = x_test.reshape((-1, 32*32*3))
    x_test_adv = attack.generate(x=x_test)

    np.save('x', x_test)
    np.save('adv_x', x_test_adv)

    print('x_test ', x_test)
    print('x_test_adv ', x_test_adv)

    # dist1 = utils.computeDist1(x_test, x_test_adv)
    # print('test data dist1: ', dist1)

    dist2 = utils.computeDist2(x_test, x_test_adv)
    print('test data dist2: ', dist2)

    distInf = utils.computeDistInf(x_test, x_test_adv)
    print('test data distInf: ', distInf)

    # avg_dist2, med_dist2 = utils.computeDist2(x_test, x_test_adv)
    # print('test avg_dist2: ', avg_dist2)
    # # print('test med_dist2: ', med_dist2)

    # avg_distInf, med_distInf = utils.computeDistInf(x_test, x_test_adv)
    # print('test avg_distInf: ', avg_distInf)
    # # print('test med_distInf: ', med_distInf)

    print('Cost time: ', time.time() - s)

attack()
