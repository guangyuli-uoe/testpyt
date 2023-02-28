import math
import torch
from torch import nn
from torch.nn import functional as F
from nlp.dataset import dataloader
import nlp.preprocess as pre

import numpy as np

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    norm = torch.sqrt(
        sum(
            torch.sum((p.grad**2)) for p in params
        )
    )

    # norm1 = sum(torch.sum(p.grad**2) for p in params)

    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    '''
        num_inputs = one_hot(X)
        X: vocab_size
    '''

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    '''
        H_t = f( X_t * W_xh 
                + H_t-1 * W_hh
                + b_h
                )
        
        shape:
            X_t: (b, d)
            W_xh: (d, h)
                b, h
            H_t: (b, h)
            W_hh: (h, h)
                b, h
            b_h: (1, h)
    '''
    '''
        O_t = H_t * W_hq + b_q
        
        shape:
            H_t: (b, h)
            W_hq: (h, q)
                b, q
            b_q: (1, q)
    '''

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rann_hidden_state(batch_size, num_hiddens, device):
    shape = (batch_size, num_hiddens)
    '''
        一些 rnn 的 隐状态 可能包含多个 变量
        所以，扔到元组里，
    '''
    return (torch.zeros(shape, device=device),)

def rnn_op(inputs, hidden_state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = hidden_state
    outputs = []
    for X in inputs:
        '''
            此处的inputs，
                是经过X.t 和 onehot编码的
                (b, t)
                    即
                (t, b)
                (t, b, v)
                
            inputs: [timestep, batch_size, vocab_size]
            for X, 会按照第一个纬度去遍历： （batch_size, vocab_size）
                所以，每轮loop其实是在算一个timestep
        '''
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        '''
            Y是当前时刻t下，预测到下一个t+1时刻是谁
        '''
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    '''
        在dim=0，即在batch—size这个纬度
        行数会变为，batch-size * num—timestep
    '''
    return torch.cat(outputs, dim=0), (H,)

class myRNN:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_hidden_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_hidden_state, self.forward_fn = init_hidden_state,forward_fn

    '''
        call  __call__(self, X, state):
        myRNN 实例化的 输入
        
        return self.forward_fn(X, state, self.params)
        输出
    '''
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_hidden_state(batch_size, self.num_hiddens, device)