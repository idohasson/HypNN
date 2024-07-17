import torch
from torch import nn

from model.layers.hyperbolic.activation import HypReLU, HypSigmoid, HypTanh, HypDropout
from model.layers.hyperbolic.hyp_utils import mobius_add, mobius_mm, mobius_prod, logmap0, expmap0, projection
from model.layers.hyperbolic.hyp_recurrent import HypRecurrent, HypRecurrentCell


class HypGRUCellHyp(HypRecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super(HypRecurrentCell, self).__init__()
        self.hidden_size = hidden_size
        k = 1 / hidden_size ** .5
        self.Wx = nn.Parameter(projection(expmap0(torch.empty(input_size, 3 * hidden_size).uniform_(-k, k))))
        self.Wh = nn.Parameter(projection(expmap0(torch.empty(hidden_size, 3 * hidden_size).uniform_(-k, k))))
        self.bx = nn.Parameter(torch.empty(1, 3 * hidden_size).zero_())
        self.bh = nn.Parameter(torch.empty(1, 3 * hidden_size).zero_())
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
        x_t = mobius_add(mobius_mm(xi, drop_Wx), self.bx)
        h_t = mobius_add(mobius_mm(state, drop_Wh), self.bh)

        x_reset, x_upd, x_new = x_t.chunk(3, -1)
        h_reset, h_upd, h_new = h_t.chunk(3, -1)

        reset_gate = mobius_add(x_reset, h_reset)
        reset_gate = torch.sigmoid(logmap0(reset_gate))

        update_gate = mobius_add(x_upd, h_upd)
        update_gate = torch.sigmoid(logmap0(update_gate))

        new_gate = mobius_prod(reset_gate, h_new)
        new_gate = mobius_add(x_new, new_gate)
        if self.activation is not None:
            new_gate = self.activation(new_gate)

        hy = mobius_add(-new_gate, state)
        hy = mobius_prod(update_gate, hy)
        hy = mobius_add(hy, new_gate)
        return hy

    def forward(self, x, hidden=None):
        return super(HypGRUCellHyp, self).forward(x, hidden)


class HypGRU(HypRecurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0, activation="tanh",
                 gradual=True):
        super(HypGRU, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual,
                                     HypGRUCellHyp)

    def forward(self, X, hidden_states=None):
        return super(HypGRU, self).forward(X, hidden_states)
