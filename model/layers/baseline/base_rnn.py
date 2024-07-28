import torch

from model.layers.baseline.recurrent import Recurrent, RecurrentCell


class RNNCell(RecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super().__init__(input_size, hidden_size, hidden_size, dropout, activation)

    def recurrent_step(self, xi, state):
        drop_Wx = torch.nn.functional.dropout(self.Wx, self.dropout, self.training)
        drop_Wh = torch.nn.functional.dropout(self.Wh, self.dropout, self.training)
        hy = torch.mm(xi, drop_Wx) + self.bx + torch.mm(state, drop_Wh) + self.bh
        return self.activation(hy) if self.activation is not None else hy

    def forward(self, x, hidden=None):
        return super(RNNCell, self).forward(x, hidden)


class RNN(Recurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0., activation="tanh",
                 gradual=True):
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual,
                                  RNNCell)

    def forward(self, X, hidden_states=None):
        return super(RNN, self).forward(X, hidden_states)
