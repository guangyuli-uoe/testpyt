import os
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

data_path = '../classify-leaves/'

def split_data(data_path, ratio=0.2):

    random.seed(0) # 保证随机结果可复线

    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    img_path = os.path.join(data_path, 'images')
    # print(train_path)


    # val_path = os.path

    '''
        
        train 18353
        test 8800
        total 27153
        ======
        train:
        val: 3670
    '''



    dataframe = pd.read_csv(train_path)
    print(dataframe.head(5))
    print(dataframe.describe())



    val = int(len(dataframe['image']) * 0.2)
    print(val)

    # testdf = pd.read_csv(test_path)
    # print(testdf.describe())

    # print(os.listdir(data_path)) # ['.DS_Store', 'images', 'test.csv', 'train.csv', 'sample_submission.csv']

    img_list = os.listdir(img_path)
    count = 0



    # if os.path.exists():




    if os.path.exists(img_path):
        print('111')
        exit()
    else:
        print('222')
        # os.mkdir()

        for img in img_list:
            if img.endswith('.jpg'):
                count += 1

    print(count)
# split_data(data_path)


def class2num(categories):
    '''

    :param labels:
    :return: dict
    把class转为num
    '''

    return dict(zip(categories, range(len(categories))))

def num2class(cate_dict):
    return {v:k for k,v in cate_dict.items()}

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'



def im_convert(tensor):
    """ 展示数据"""
    print(tensor.shape) # torch.Size([3, 224, 224])
    image = tensor.to("cpu").clone().detach()
    print(image.shape) # torch.Size([3, 224, 224])
    image = image.numpy().squeeze()
    print(image.shape) # (3, 224, 224)
    print(type(image)) # <class 'numpy.ndarray'>
    image = image.transpose(1, 2, 0)  # TypeError: Invalid shape (3, 224, 224) for image data
    image = image.clip(0, 1)

    return image


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


def draw_acc_loss(train_acc_list, train_loss_list, val_acc_list, val_loss_list,num_epoch):
    x = np.arange(num_epoch)
    plt.style.use('ggplot')
    '''
        自定义常用参数
    '''
    # 设置支持中文字体（黑体）
    # mpl.rcParams['font.family'] = ['Heiti SC']
    # 提高图片清晰度, dots per inch
    # mpl.rcParams['figure.dpi'] = 300
    fig, axes = plt.subplots(2, 2,
                             # facecolor='gray', # 设置背景颜色
                             # sharex=True,
                             # sharey=True
                             )
    axes[0, 0].plot(x, train_acc_list, label='train-acc')
    axes[0, 0].plot(x, train_loss_list, label='train-loss')
    # axes[0, 0].set_title('train')
    axes[0, 0].set_xlabel('epoch')
    axes[0,0].set_ylabel('train')

    axes[1, 0].plot(x, val_acc_list, label='val-acc')
    axes[1, 0].plot(x, val_loss_list, label='val-loss')
    # axes[1, 0].set_title('valid')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('valid')

    axes[0, 1].plot(x, train_acc_list, label='train-acc')
    axes[0, 1].plot(x, val_acc_list, label='val-acc')
    # axes[1, 0].set_title('train-valid')
    axes[0, 1].set_xlabel('epoch')
    # axes[1, 0].set_ylabel('acc')
    # plt.legend()

    axes[1, 1].plot(x, train_loss_list, label='train-loss')
    axes[1, 1].plot(x, val_loss_list, label='val-loss')
    # axes[1, 1].set_title('train-valid')
    axes[1, 1].set_xlabel('epoch')
    # axes[1, 1].set_ylabel('loss')

    plt.grid()
    plt.legend()
    plt.savefig('result.pdf')
    plt.show()


# if __name__ == '__main__':
#     train_acc = np.linspace(0, 1, 50)
#     train_loss = np.linspace(10, 100, 50)
#
#     # val_acc = np.random.randint(0, 1, size=50)
#     val_acc = np.random.sample(50)
#     val_loss = np.random.randint(1, 100, size=50)
#
#     draw_acc_loss(train_acc, train_loss, val_acc, val_loss, 50)
