from mykaggle.dataset import mydataset, utils
import torch
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    csvpath = '../classify-leaves/newtrain.csv'
    imgpath = '../classify-leaves/'
    # valcsvpath = '../claasify-leaves/val.csv'
    test_path = '../classify-leaves/newtest.csv'

    train_dataset = mydataset.DemoDataset(csvpath,imgpath,mode='train')
    print(train_dataset)
    print(len(train_dataset))

    test_dataset = mydataset.DemoDataset(test_path, imgpath, mode='test')
    print(test_dataset)
    print(len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        # num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        # num_workers=5
    )


    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()

    img = train_features[0]
    img = utils.im_convert(img)

    label = train_labels[0]
    print(label)
    print(type(label))
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}, {train_dataset.num2class_dict[int(label)]}")