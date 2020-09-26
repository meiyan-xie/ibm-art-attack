import numpy as np


def concat_adv_resnet():
    path = 'resnet_adv_data_5/'
    adv_data = []
    adv_path = [path + 'resnet_adv_data_%d.npy' % i for i in range(10)]
    for i in range(10):
        adv_data.append(np.load(adv_path[i]))

    adv_data = np.concatenate(adv_data, axis=0)
    print(adv_data.shape)
    np.save('resnet_adv_data_5', adv_data)


concat_adv_resnet()



def concat_adv_bnn():
    path = 'bnn_adv_data_5/'

    adv_data = []
    adv_path = [path + 'bnn_adv_data_%d.npy' % i for i in range(100)]

    for i in range(100):
        adv_data.append(np.load(adv_path[i]))

    adv_data = np.array(adv_data)
    adv_data = np.squeeze(adv_data, axis=1)
    print(adv_data.shape)
    np.save('bnn_adv_data_5', adv_data)

# concat_adv_bnn()
