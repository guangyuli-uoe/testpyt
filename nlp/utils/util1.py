import torch
from nlp.dataset import dataloader

class seqDataloader:
    def __init__(self, corpus, batch_size, num_steps, mode):
        self.corpus = corpus
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.mode = mode
        self.loader = dataloader.my_iter_random
    def __iter__(self):
        return self.loader(self.corpus, self.batch_size, self.num_steps, self.mode)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.
    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()