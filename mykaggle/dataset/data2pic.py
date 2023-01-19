from mykaggle.dataset import mydataset, utils
import torch
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    csvpath = '../classify-leaves/newtrain.csv'
    imgpath = '../classify-leaves/'

    train_dataset = mydataset.DemoDataset(csvpath,imgpath,mode='train')
    print(train_dataset)
    print(len(train_dataset))
    # print(train_dataset[0])

    # print(utils.get_device())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        # num_workers=5
    )
    # print(train_loader)

    for img,label in train_loader:
        # print(img)
        print(label)
        break

    # fig = plt.figure(figsize=(20, 12))
    fig = plt.figure()
    columns = 4
    rows = 2

    dataiter = iter(train_loader)
    print(type(dataiter)) # <class 'torch.utils.data.dataloader._SingleProcessDataLoaderIter'>
    print()
    inputs, classes = next(dataiter)
    print(type(inputs)) # <class 'torch.Tensor'>
    print(inputs.shape) # torch.Size([8, 3, 224, 224])

    print('===============================')

    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.set_title(train_dataset.num2class_dict[int(classes[idx])])
        plt.imshow(utils.im_convert(inputs[idx]))
        print(type(inputs[idx]))
        '''
            im_convert is necessary
        '''
        # plt.imshow(inputs[idx].transpose(1, 2, 0))

        break
    # plt.show()