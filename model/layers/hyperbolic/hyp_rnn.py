import torch
from torch import nn

from model.layers.hyperbolic.hyp_utils import mobius_add, mobius_mm, expmap0, projection
from model.layers.hyperbolic.hyp_recurrent import HypRecurrent, HypRecurrentCell
from model.layers.hyperbolic.activation import HypReLU, HypSigmoid, HypTanh, HypDropout

class HypRNNCellHyp(HypRecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super(HypRecurrentCell, self).__init__()
        self.hidden_size = hidden_size
        k = 1 / hidden_size ** .5
        self.Wx = nn.Parameter(projection(expmap0(torch.empty(input_size, hidden_size).uniform_(-k, k))))
        self.Wh = nn.Parameter(projection(expmap0(torch.empty(hidden_size, hidden_size).uniform_(-k, k))))
        self.bx = nn.Parameter(torch.empty(1, hidden_size).zero_())
        self.bh = nn.Parameter(torch.empty(1, hidden_size).zero_())
        self.dropout = HypDropout(dropout)
        if activation == 'tanh':
            self.activation = HypTanh()
        elif activation == 'relu':
            self.activation = HypReLU()
        elif activation == 'sigmoid':
            self.activation = HypSigmoid()
        else:
            self.activation = None

    def recurrent_step(self, xi, state):
        drop_Wx = self.dropout(self.Wx, self.training)
        drop_Wh = self.dropout(self.Wh, self.training)
        hy = mobius_add(
            mobius_add(mobius_mm(xi, drop_Wx), self.bx),
            mobius_add(mobius_mm(state, drop_Wh), self.bh)
        )
        return self.activation(hy) if self.activation is not None else hy

    def forward(self, x, hidden=None):
        return super(HypRNNCellHyp, self).forward(x, hidden)


class HypRNN(HypRecurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0., activation="tanh", gradual=True):
        super(HypRNN, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual, HypRNNCellHyp)

    def forward(self, X, hidden_states=None):
        return super(HypRNN, self).forward(X, hidden_states)
