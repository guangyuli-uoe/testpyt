import os

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


from mykaggle.dataset import utils

class DemoDataset(Dataset):
    def __init__(self, csv_path, img_path, mode):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.data_len = len(self.data_info.index) - 1
        self.img_path = img_path
        self.mode = mode
        # print('1111111111111')

        if mode == 'train':
            '''
                array和asarray都可以将结构数据转化为ndarray，
                但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会。
                
                iloc[ : , : ]
                前面的冒号就是取行数，后面的冒号是取列数
                左闭右开原则
                
            '''
            self.img_list = np.asarray(self.data_info.iloc[1:len(self.data_info.index), 0])
            self.label_list = np.asarray(self.data_info.iloc[1:len(self.data_info.index), 1])
            self.classes = sorted(list(set(self.label_list)))
            self.class2num_dict = utils.class2num(self.classes)
            self.num2class_dict = utils.num2class(self.class2num_dict)

        elif mode == 'test':
            self.img_list = np.asarray(self.data_info.iloc[1:len(self.data_info.index), 0])

        print(f'the {self.mode} dataset: {len(self.img_list)} samples, ')

    def __getitem__(self, item):
        # item == index
        single_image_name = self.img_list[item]
        img = Image.open(os.path.join(self.img_path, single_image_name))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img = transform(img)

        if self.mode == 'test':
            return img
        else:
            label = self.label_list[item]
            num_label = self.class2num_dict[label]

            return img, num_label


    def __len__(self):
        return len(self.img_list)








