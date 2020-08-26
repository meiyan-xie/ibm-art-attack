import numpy as np
from numpy import linalg as LA

# Load the STL10 dataset
def loadData():
    trainfile = 'stl10_data/bsptrain.0'
    train = np.loadtxt(trainfile)
    x_train = train[:, 1:]
    y_train = train[:, 0]
    y_train[y_train==-1] = 0
    x_train = x_train / 255
    print('x_train shape', x_train.shape)

    testfile = 'stl10_data/bsptest.0'
    test = np.loadtxt(testfile)
    x_test = test[:, 1:]
    y_test = test[:, 0]
    y_test[y_test==-1] = 0
    x_test = x_test / 255
    print('x_test shape', x_test.shape)
    print('x_test ', x_test[-1])

    return x_train, y_train, x_test, y_test


# Randomly select n samples for per class to generate adversarial sample
def randomlySelect(num, y):
        pos_indices = []
        neg_indices = []
        for i in range(num):
            if y[i] == 0:
                neg_indices.append(i)
            elif y[i] == 1:
                pos_indices.append(i)

        # print('pos_indices ', pos_indices[0])
        # print('neg_indices ', neg_indices[0])
        np.random.seed(2019)
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)
        print('pos_indices ', pos_indices[0])
        print('neg_indices ', neg_indices[0])

        return pos_indices[0], neg_indices[0]


# Compute distance between clean and adv data
def computeDist1(x_clean, x_adv):
    dist = LA.norm(x_clean - x_adv, ord=1, axis=1)
    # Average dist of all the samples
    dist = np.sum(dist) / len(x_clean)

    return dist

def computeDist2(x_clean, x_adv):
    dist = LA.norm(x_clean - x_adv, ord=2, axis=1)
    # Average dist of all the samples
    dist = np.sum(dist) / len(x_clean)

    return dist

def computeDistInf(x_clean, x_adv):
    dist = LA.norm(x_clean - x_adv, ord=np.inf, axis=1)
    # Average dist of all the samples
    dist = np.sum(dist) / len(x_clean)

    return dist
