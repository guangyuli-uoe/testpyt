import math
import torch
from torch import nn
from torch.nn import functional as F
from nlp.dataset import dataloader
import nlp.preprocess as pre

from nlp.rnn import rnn111
from nlp.utils import util1
import matplotlib.pyplot as plt
import numpy as np

import sys



def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    print('==============  training epoch ==============')
    state = None
    # metric = util1.Accumulator(2) # 训练损失之和，词元数量
    # count = 0
    epoch_ppl = []
    for i, (X, Y) in enumerate(train_iter):
        # count += 1
        '''
            X,Y [batch_size, num_step]
        '''
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else: # state != None and dont use random
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        '''
            reshape(-1): 改成一串，没有行列
            reshape(-1， 1)：改成一列，不知道几行
        '''
        y = Y.T.reshape(-1)
        '''
            (b, t): (3, 5)
            (5, 3)
            
            (t1_1, t1_2, t1_3,
                ...
            t5_1, t5_2, t5_3)
            
            y, (bt,)
        '''
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        '''
            y_hat, (b*t, v)
                pytorch, .long() 是向下取整
        '''
        l = loss(y_hat, y.long()).mean()
        ppl = math.exp(l)
        epoch_ppl.append(ppl)
        # print(ppl)
        # l_1 = loss(y_hat, y.long())
        # print(f'l: {l}, y.numel: {y.numel()}, l_1: {l_1}')
        # print(f'y_long: {y.long()}')
        # print(f'y: {y}')

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            rnn111.grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            rnn111.grad_clipping(net, 1)
            # 因为已经调用列mean函数
            updater(batch_size=1)
        # metric.add(l*y.numel(), y.numel())
        # print(f'batch: {count}')
    # print(len(epoch_ppl))
        print(f'[{i} batch], loss: {l}, ppl: {ppl}')
    print('*****************  training epoch over ******************' + '\n')
    return sum(epoch_ppl)/len(epoch_ppl), len(epoch_ppl)


def predict_ch8(prefix, num_preds, net, vocab, device):

    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    '''
        get_input:
            将outputs里最后一个变成tensor，（b, t）
    '''
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1)) # (batch_size, time_step)
    for y in prefix[1:]:
        # print(f'get_input: {get_input()}')
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
        # vocab[y] 是 index
        # print(f'vocab[y]: {vocab[y]}')
        # print(outputs)

    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # print(f'y: {y}')
        # print(f'y.shape: {y.shape}') # (1, 28)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])


def draw(ppl, num_epochs):
    x = np.arange(num_epochs)
    plt.style.use('ggplot')
    '''
        自定义常用参数
    '''
    # 设置支持中文字体（黑体）
    # mpl.rcParams['font.family'] = ['Heiti SC']
    # 提高图片清晰度, dots per inch
    # mpl.rcParams['figure.dpi'] = 300
    fig, axes = plt.subplots(1, 1,
                             # facecolor='gray', # 设置背景颜色
                             # sharex=True,
                             # sharey=True
                             )
    axes.plot(x, ppl,label='train')
    axes.set_xlabel('epoch')
    axes.set_ylabel('ppl')

    plt.grid()
    plt.legend()
    plt.savefig('result.pdf')
    plt.show()

def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=True):

    loss = nn.CrossEntropyLoss()

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: util1.sgd(net.params, lr, batch_size=batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    # 训练 和 预测
    print('=========  training  ===========')
    ppl_list = []
    loss_list = []

    for epoch in range(num_epochs):
        avg_ppl_per_batch, num_ppl_per_batch = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        # train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        ppl_list.append(avg_ppl_per_batch)
        # loss_list.append(loss)
        print(f'epoch: {epoch+1}, avg_ppl: {avg_ppl_per_batch}, num_ppl_per_batch: {num_ppl_per_batch}')

        print('------------  predicting  ---------------')
        print(predict('time traveller '))
        print('------------  predicting over  ---------------')

    print('=========  training  over  ===========')
    draw(ppl_list, num_epochs)

log_print = open('../log/1.log', 'w')
sys.stdout = log_print
sys.stderr = log_print

if __name__ == '__main__':
    # metric = util1.Accumulator(2)
    # print(metric)
    # print(metric.data)
    # a,b,c = 1,2,2
    # metric.add(a, b,c)
    # print(metric.data)
    num_hiddens = 256
    batch_size, num_steps = 32, 35
    num_epochs = 56
    lr = 0.01

    txtpath = '../timemachine.txt'
    corpus, vocab = pre.load_corpus_tm(txtpath, mode='char')
    print(len(vocab))

    # data_iter = dataloader.my_iter_random(corpus, batch_size=batch_size, num_steps=num_steps, mode='random')

    data_iter = util1.seqDataloader(corpus, batch_size, num_steps, 'random')

    net = rnn111.myRNN(len(vocab), num_hiddens, rnn111.get_device(), rnn111.get_params,
                       rnn111.init_rann_hidden_state, rnn111.rnn_op)

    train(net, data_iter, vocab, lr, num_epochs, device=rnn111.get_device(), use_random_iter=True)


    '''
    
        batch_size: 32
        num_subseqs: 174701
        len(new_split_corpus2): 174701
        num_batches: 5459
        18.751459377288967
    '''
