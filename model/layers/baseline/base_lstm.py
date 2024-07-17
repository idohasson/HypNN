import torch
from torch import nn

from model.layers.baseline.recurrent import Recurrent, RecurrentCell


class LSTMCell(RecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super(RecurrentCell, self).__init__()
        self.hidden_size = hidden_size
        k = 1 / hidden_size ** .5
        self.Wx = nn.Parameter(torch.empty(input_size, 4 * hidden_size).uniform_(-k, k))
        self.Wh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size).uniform_(-k, k))
        self.bx = nn.Parameter(torch.empty(1, 4 * hidden_size).zero_())
        self.bh = nn.Parameter(torch.empty(1, 4 * hidden_size).zero_())
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
        if not isinstance(state, tuple):
            state = (state, state)
        hx, cx = state
        drop_Wx = torch.nn.functional.dropout(self.Wx, self.dropout, self.training)
        drop_Wh = torch.nn.functional.dropout(self.Wh, self.dropout, self.training)
        gates = torch.mm(xi, drop_Wx) + self.bx + torch.mm(hx, drop_Wh) + self.bh
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, -1)
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = self.activation(cell_gate) if self.activation is not None else cell_gate
        o_t = torch.sigmoid(output_gate)
        cy = cx * f_t + i_t * g_t
        cy = self.activation(cy) if self.activation is not None else cy
        hy = o_t * cy
        return hy, cy

    def forward(self, x, hidden=None):
        return super(LSTMCell, self).forward(x, hidden)


class LSTM(Recurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=.0, activation="tanh",
                 gradual=True):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual,
                                      LSTMCell)

    def forward(self, X, hidden_states=None):
        return super(LSTM, self).forward(X, hidden_states)
