import torch
from torch import nn
from torch.nn import functional as F

# batch_size, num_step = 32, 35
# num_hiddens = 256

# rnn_layer = nn.RNN()

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)

        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 此时，Y （t, b, h）
        # state (1/2, b, h)

        '''
            全连接层 首先将 Y的形状改为 （timestep * batch_size, num_hidden）
            他的输出是 （timestep * batch_size, len(vocab)）
        '''
        output = self.linear(Y.reshape(
            (-1, Y.shape[-1]) # (tb, h) * (h, v)
        ))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU 以 张量 作为 隐状态
            return torch.zeros(
                (self.num_hiddens * self.rnn.num_layers,
                 batch_size,
                 self.num_hiddens),
                device=device
            )
        else:
            # nn.LSTM 以 元组 作为隐状态
            return (
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,
                     batch_size,
                     self.num_hiddens),
                    device=device
                ),

                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,
                     batch_size,
                     self.num_hiddens),
                    device=device
                )
            )
