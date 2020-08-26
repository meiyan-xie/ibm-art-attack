import numpy as np
from PIL import Image


adv_rf = np.load('xtrain.npy')
# adv_scd = np.load('x_test_adv_iter50_1sample_scd.npy')
# adv_svc = np.load('x_test_adv_iter50_1sample_svc.npy')
# org = np.loadtxt('../../stl10_data/bsptest.0')

# adv_rf =  (adv_rf * 255).astype('uint8').reshape((-1, 3, 48, 48))[0]

adv_rf = (adv_rf * 255).astype('uint8').reshape((-1, 3, 48, 48))[0]
adv_rf = np.transpose(adv_rf, (1, 2, 0))

# adv_scd =  (adv_scd * 255).astype('uint8').reshape((-1, 96, 96, 3))[0]
# adv_svc =  (adv_svc * 255).astype('uint8').reshape((-1, 96, 96, 3))[0]
# org = org[:, 1:][0].astype('uint8').reshape((-1, 96, 96, 3))[0]

print(adv_rf.shape)
# print(adv_scd.shape)
# print(adv_svc.shape)
# print(org.shape)

im_rf = Image.fromarray(adv_rf, mode='RGB')
# im_scd = Image.fromarray(adv_scd, mode='RGB')
# im_svc = Image.fromarray(adv_svc, mode='RGB')
# im_org = Image.fromarray(org, mode='RGB')

im_rf.save('x.jpeg')
# im_scd.save('adv_test0_iter50_scd.jpeg')
# im_svc.save('adv_test0_iter50_svc.jpeg')
# im_org.save('org_test0.jpeg')
