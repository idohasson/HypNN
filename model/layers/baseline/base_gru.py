import torch

from model.layers.baseline.recurrent import Recurrent, RecurrentCell


class GRUCell(RecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super().__init__(input_size, hidden_size, 3 * hidden_size, dropout, activation)

    def recurrent_step(self, xi, state):
        drop_Wx = torch.nn.functional.dropout(self.Wx, self.dropout, self.training)
        drop_Wh = torch.nn.functional.dropout(self.Wh, self.dropout, self.training)
        x_t = torch.mm(xi, drop_Wx) + self.bx
        h_t = torch.mm(state, drop_Wh) + self.bh
        x_reset, x_upd, x_new = x_t.chunk(3, -1)
        h_reset, h_upd, h_new = h_t.chunk(3, -1)
        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = x_new + (reset_gate * h_new)
        new_gate = self.activation(new_gate) if self.activation is not None else new_gate
        hy = update_gate * state + (1 - update_gate) * new_gate
        return hy

    def forward(self, x, hidden=None):
        return super(GRUCell, self).forward(x, hidden)


class GRU(Recurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=.0, activation="tanh",
                 gradual=True):
        super(GRU, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual,
                                  GRUCell)

    def forward(self, X, hidden_states=None):
        return super(GRU, self).forward(X, hidden_states)
