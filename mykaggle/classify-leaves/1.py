import pandas as pd
import numpy as np
from PIL import Image
import os


path = './newtrain.csv'
# path = './test.csv'
# path = './train.csv'


dataframe = pd.read_csv(path, header=None)
print(dataframe.head(5))
print()

# classes = list(set(dataframe['label']))
# print(classes)

# print(len(dataframe))
#
print(dataframe.index)
print(f'len(dataframe.index): {len(dataframe.index)}')
#
imgs = np.asarray(dataframe.iloc[1:len(dataframe.index), 0])
# print(imgs)
print(len(imgs))
print(type(imgs))
print(imgs[0])

single_img = Image.open(os.path.join('./', imgs[0]))
print(single_img)

# print(dataframe.iloc[1:len(dataframe.index), 0])

'''
    for labels
'''
labels = np.asarray(dataframe.iloc[1:len(dataframe.index), 1])
# print(dataframe.iloc[1:len(dataframe.index), 1])
print(len(labels))

classes = sorted(list(set(labels)))
print(len(classes))
# print(f'set: {len(set(labels))}')

def num2class(cate_dict):
    return {v:k for k,v in cate_dict.items()}

def class2num(categories):
    '''

    :param labels:
    :return: dict
    把class转为num
    '''

    return dict(zip(categories, range(len(categories))))

print(class2num(classes))

print(num2class(class2num(classes)))
