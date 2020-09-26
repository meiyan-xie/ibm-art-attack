import sys
import numpy as np
import pickle

sys.path.append('..')
from core.scd_lib import *
from tools import args, save_checkpoint, print_title, load_data
from core.bnn import *


def main():

    # # Define variable
    # modelpath = '../binary/checkpoints/cifar10_scdcemlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl'
    # print('\n------------- model -------------\n', modelpath)

    # Load model
    # with open(modelpath, 'rb') as f:
    #     model = pickle.load(f)

    model = BNN(['../binary/checkpoints/cifar10_mlpbnn_approx_%d.h5' % (i) for i in range(100)])
    print('------------- model -------------\n', 'cifar10_mlpbnn_approx')

    adv_data = np.load('bnn_adv_data_5.npy')
    print('adv shape', adv_data.shape)
    # adv_data = adv_data.reshape(-1, 3, 32, 32)
    # print('adv shape', adv_data.shape)

    num_vote = 100
    misclassified_rate_lst = []

    for vote in range(num_vote):
        print('=======================')
        print('\nVote id: {}\n'.format(vote))
        misclassified_count = 0
        pred_y = model.predict(adv_data, best_index=vote).astype(int)
        # pred_y = model.predict(adv_data, kind='best', best_index=vote).astype(int)
        # pred_y = model.estimators_[vote].predict(adv_data)

        # The true label is 0
        print('pred_y', pred_y)
        for idx in range(len(pred_y)):
            if pred_y[idx] == 1:
                misclassified_count += 1

        print('Vote {} misclassified count is {}\n'.format(vote, misclassified_count))

        misclassified_rate = misclassified_count / num_vote
        misclassified_rate_lst.append(misclassified_rate)

    misclassified_rate_arr = np.array(misclassified_rate_lst)
    avg_rate = np.mean(misclassified_rate_arr)

    print('\nAverage misclassified rate across 100 models is {}\n'.format(avg_rate))

main()
#
