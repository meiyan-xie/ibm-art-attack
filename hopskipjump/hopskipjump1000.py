import time
import numpy as np
from art.attacks.evasion import HopSkipJump
from art.classifiers import BlackBoxClassifier

import utils


def distortion(original_data, adv_data):
    '''
    Compute the distance between original data and adversary data
    '''
    print('x ', original_data)
    print('x_adv ', adv_data)

    dist2 = utils.computeDist2(original_data, adv_data)
    print('test data dist2: ', dist2)

    distInf = utils.computeDistInf(original_data, adv_data)
    print('test data distInf: ', distInf)


def attack(predictWrapper, x_train, x_test, y_train, y_test, input_shape, datapoint):

    min_pixel_value = x_train.min()
    max_pixel_value = x_train.max()
    print('min_pixel_value ', min_pixel_value)
    print('max_pixel_value ', max_pixel_value)

    print('xtrain shape: ', x_train.shape)
    print('xtest shape: ', x_test.shape)
    print('y_train shape: ', y_train.shape)
    print('ytest shape: ', y_test.shape)

    # Create classifier
    classifier = BlackBoxClassifier(predict=predictWrapper.predict_one_hot,
                                    input_shape=(input_shape, ),
                                    nb_classes=2,
                                    clip_values=(min_pixel_value, max_pixel_value))

    print('----- generate adv data by HopSkipJump attack -----')
    # Generate adversarial test examples
    s = time.time()

    attacker = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=100, max_eval=10000, init_eval=100, init_size=100)
    # attacker = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=2, max_eval=10000, init_eval=100, init_size=100)


    # Input data shape should be 2D
    datapoint = datapoint.reshape((-1, input_shape))
    adv_data = attacker.generate(x=datapoint)

    distortion(datapoint, adv_data)
    print('Generate test adv cost time: ', time.time() - s)

    return adv_data
