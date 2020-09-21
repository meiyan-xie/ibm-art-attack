import numpy as np


def concat_adv_resnet():
    path = 'resnet_adv_data_288/'
    adv_data = []
    adv_path = [path + 'resnet_adv_data_%d.npy' % i for i in range(10)]
    for i in range(10):
        adv_data.append(np.load(adv_path[i]))

    adv_data = np.concatenate(adv_data, axis=0)
    print(adv_data.shape)
    np.save('resnet_adv_data_288', adv_data)


concat_adv_resnet()
