import numpy as np


bnn_adv_data = []

for i in range(100):
    adv = np.load('bnn_adv_data/bnn_adv_data_{}.npy'.format(i))
    print(adv.shape)

    bnn_adv_data.append(adv)

bnn_adv_data = np.array(bnn_adv_data)
bnn_adv_data = np.squeeze(bnn_adv_data, axis=1)
print(bnn_adv_data.shape)
np.save('bnn_adv_data', bnn_adv_data)
