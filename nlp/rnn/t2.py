import math
import torch
from torch import nn
from torch.nn import functional as F
from nlp.dataset import dataloader
import nlp.preprocess as pre

from nlp.rnn import rnn111


num_hiddens = 512
batch_size, num_steps = 32, 35

txtpath = '../timemachine.txt'
corpus, vocab = pre.load_corpus_tm(txtpath, mode='char')
print(len(vocab))

data_iter = dataloader.my_iter_random(corpus, batch_size=batch_size, num_steps=num_steps, mode='random')

# for X,Y in data_iter:
#     print(f'X: {X}')
#     print(f'Y: {Y}')

device = rnn111.get_device()
# params = rnn111.get_params(len(vocab), num_hiddens, device)
# init_hidden = rnn111.init_rann_hidden_state(batch_size, num_hiddens, device)

net = rnn111.myRNN(len(vocab), num_hiddens, rnn111.get_device(), rnn111.get_params,
                   rnn111.init_rann_hidden_state, rnn111.rnn_op)

X = torch.arange(10).reshape((2, 5))
'''
    (2, 5)
    (batch_size, timestep)
'''
print(f'X.shape: {X.shape}')
state = net.begin_state(X.shape[0], rnn111.get_device())
Y, new_state = net(X.to(device), state)
'''
    new_state: tuple
'''
print(f'Y.shape: {Y.shape}')
'''
    y, (b*t, v)
    
    y, (t*b, v)
'''
print(len(Y)) # b: 10
print(len(new_state)) # 1
print(new_state[0].shape)
# print(new_state.shape)


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

print(predict_ch8('time', 100, net, vocab, device))

