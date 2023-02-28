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
print(vocab)
print(vocab.idx_to_token)
print(len(vocab.idx_to_token))
print(vocab['e'])
print(vocab['q'])
print(vocab.token_to_idx)
print(vocab[1])