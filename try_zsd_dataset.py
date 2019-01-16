from src.zsd_dataset import ZSD_Dataset
import os
import argparse
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.utils import *

from src.ZSD_loss import ZSDLoss
from src.zsd_net import ZSD
from train_voc import get_args

opt = get_args()
training_params = {"batch_size": opt.batch_size,
                       "shuffle": False,
                       "drop_last": True,
                       "collate_fn": custom_collate_fn}

training_set = ZSD_Dataset(opt.data_path,mode='seen', image_size=opt.image_size)
training_generator = DataLoader(training_set, **training_params)
# for iter, batch in enumerate(training_generator):
#     img,label = batch # [1,3,448,448]
#     print('img.shape',img.shape)
#     print(label[0][0][5])

# import cv2
# img = cv2.imread('/home/neec10601/Data/hmb/ZSD/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg')
# print(img.shape)

attr = np.load('/home/neec10601/Data/hmb/ZSD/attributes/attrs.pkl.npy')
seen = np.ndarray((10,64))
k = [1,2,4,5,8,10,11,14,15,19]
for i in range(10):
    seen[i] = attr[k[i]]
print(seen)
np.save('/home/neec10601/Data/hmb/ZSD/attributes/seen.pkl',seen)