from mykaggle.dataset import mydataset, utils
import torch
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    csvpath = '../classify-leaves/newtrain.csv'
    imgpath = '../classify-leaves/'

    # csvpath = ''

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
    #
    #     for img,label in train_loader:
    #         # print(img)
    #         print(label)
    #         breakprint(train_loader)

    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

