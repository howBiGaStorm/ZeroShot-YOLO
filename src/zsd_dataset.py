"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *
import numpy as np


class ZSD_Dataset(Dataset):
    def __init__(self, root_path="data/1010split", mode="seen", image_size=448, is_training = True):

        if (mode in ["mix", "seen", "test_seen", "unseen",'try'] ) :
            self.data_path = os.path.join(root_path, mode+'.txt')
        self.pic_paths = [path for path in open(self.data_path)]
        self.num_images = len(self.pic_paths)

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.attrs = np.load('/home/neec10601/Data/hmb/ZSD/attributes/attrs.pkl.npy')
        self.num_classes = len(self.classes)

        self.image_size = image_size
        self.is_training = is_training



    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        pic_path = self.pic_paths[item].strip()
        img = cv2.imread(pic_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp = pic_path.split('JPEGImages')
        image_xml_path = temp[0] + 'Annotations' + temp[1].split('jpg')[0] + 'xml'

        annot = ET.parse(image_xml_path)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            attr = self.attrs[label]
            # print(label)
            objects.append([xmin, ymin, xmax, ymax, label,attr])


        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), objects #np.array(objects, dtype=np.float32)
