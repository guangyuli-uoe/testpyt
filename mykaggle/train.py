import torch
import torch.nn as nn
import torch.nn.functional as F
from mykaggle.model import mycnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mykaggle.dataset import mydataset, utils
# import torch
from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys

'''
    .data
        返回的是一个tensor
    .item
        返回的是一个具体的值
    对元素不止一个对tensor列表，使用item()会报错
'''

learning_rate = 3e-4
weight_decay = 1e-3
num_epoch = 50
in_channel = 3
out_channel = 10
model_path = './output/cnn1_model.ckpt'
batchsize = 10


model = mycnn.DemoCNN(in_channel=3, out_channel=10)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

'''
    loss and optimizer

    crossEntropy
        softmax -> log -> negative log likelihood
'''
criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)




def mytrain(trainloader, current_epoch):

    # for epoch in range(epochs):
    model.train()
    train_loss = []
    train_acc1 = []
    train_acc2 = []

    for batch in tqdm(trainloader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs)
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        # Compute the gradients for parameters.
        loss.backward()
        # Update the parameters with computed gradients.
        optimizer.step()

        acc1 = (logits.argmax(dim=-1) == labels).float().mean()
        rights, l = utils.accuracy(logits, labels)
        acc2 = rights/l

        train_loss.append(loss.item())
        train_acc1.append(acc1)
        train_acc2.append(acc2)

    avg_loss = sum(train_loss) / len(train_loss)
    avg_acc1 = sum(train_acc1) / len(train_acc1)
    avg_acc2 = sum(train_acc2) / len(train_acc2)

    print(f'[---train--- epoch: {current_epoch+1}], avg_acc_1: {avg_acc1}, acg_acc_2: {avg_acc2}, avg_loss: {avg_loss}')

    return avg_loss, avg_acc1



def mytest(testloader, current_epoch):
    valid_loss = []
    valid_accs = []
    for batch in tqdm(testloader):
        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    avg_acc = sum(valid_accs) / len(valid_accs)
    avg_loss = sum(valid_loss) / len(valid_loss)

    print(f'[---valid--- epoch: {current_epoch+1}], test_avg_acc: {avg_acc}, test_avg_loss: {avg_loss}')
    return avg_acc, avg_loss

log_print = open('./log/1.log', 'w')
sys.stdout = log_print
sys.stderr = log_print

if __name__ == '__main__':
    csvpath = './classify-leaves/newtrain.csv'
    imgpath = './classify-leaves/'
    valcsvpath = './claasify-leaves/val.csv'
    test_path = './classify-leaves/newtest.csv'
    valpath = './classify-leaves/val.csv'

    train_dataset = mydataset.DemoDataset(csvpath, imgpath, mode='train')
    print(train_dataset)
    print(len(train_dataset))

    test_dataset = mydataset.DemoDataset(valpath, imgpath, mode='train')
    print(test_dataset)
    print(len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        # num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        # num_workers=5
    )

    # train_features, train_labels = next(iter(train_loader))

    print('====== start training ======')

    # print(train_features.size())
    # print(model(train_features).size()) # logits, torch.Size([8, 10])

    # acc = (logits.argmax(dim=-1) == labels).float().mean()
    # a = model(train_features).argmax(dim=-1)
    # print(a.size())
    # print(model(train_features))
    # print(a)
    # print()

    # for batch in train_loader:
    #     imgs, labels = batch
    #     print(imgs.size())
    #     print(labels.size())
    #     logits = model(imgs)
    #     print(logits)
    #     print(logits.argmax(dim=-1))
    #     print(torch.max(logits.data, 1)[1])
    #     # print(torch.max(logits, 1)[1])
    #     print(f'labels: {labels}')
    #     acc1 = (logits.argmax(dim=-1) == labels).float().mean()
    #     print(f'acc1: {acc1}')
    #     rights = torch.max(logits.data, 1)[1].eq(labels.data.view_as(torch.max(logits.data, 1)[1])).sum()
    #     print(f'rights: {rights}')
    #     break

    best_acc = 0.0

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(num_epoch):

        train_avg_loss, train_avg_acc1 = mytrain(train_loader, epoch)
        print('====== performance on valid dataset ======')
        valid_avg_acc, valid_avg_loss = mytest(test_loader, epoch)
        train_acc_list.append(train_avg_acc1)
        train_loss_list.append(train_avg_loss)
        val_acc_list.append(valid_avg_acc)
        val_loss_list.append(valid_avg_loss)

        if valid_avg_acc > best_acc:
            best_acc = valid_avg_acc
            torch.save(model.state_dict(), model_path)
            print(f'saving model..., with best_acc: {best_acc}')

    # for batch in tqdm(test_loader):
    #     imgs, labels = batch
    #     print(imgs.size())
    #     print(labels.size())
    #     break

    print(train_acc_list)
    print(train_loss_list)
    print(val_acc_list)
    print(val_loss_list)

    print('====== start drawing ======')
    utils.draw_acc_loss(train_acc_list, train_loss_list, val_acc_list, val_loss_list, num_epoch)

