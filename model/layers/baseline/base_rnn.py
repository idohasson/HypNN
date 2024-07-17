import torch
from torch import nn

from model.layers.baseline.recurrent import Recurrent, RecurrentCell


class RNNCell(RecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super(RecurrentCell, self).__init__()
        self.hidden_size = hidden_size
        k = 1 / hidden_size ** .5
        self.Wx = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-k, k))
        self.Wh = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-k, k))
        self.bx = nn.Parameter(torch.empty(1, hidden_size).zero_())
        self.bh = nn.Parameter(torch.empty(1, hidden_size).zero_())
        self.dropout = dropout
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def recurrent_step(self, xi, state):
        drop_Wx = torch.nn.functional.dropout(self.Wx, self.dropout, self.training)
        drop_Wh = torch.nn.functional.dropout(self.Wh, self.dropout, self.training)
        hy = torch.mm(xi, drop_Wx) + self.bx + torch.mm(state, drop_Wh) + self.bh
        return self.activation(hy) if self.activation is not None else hy

    def forward(self, x, hidden=None):
        return super(RNNCell, self).forward(x, hidden)


class RNN(Recurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0., activation="tanh", gradual=True):
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual, RNNCell)

    def forward(self, X, hidden_states=None):
        return super(RNN, self).forward(X, hidden_states)
