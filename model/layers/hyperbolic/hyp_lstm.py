import torch

from model.layers.hyperbolic.hyp_recurrent import HypRecurrent, HypRecurrentCell
from model.layers.hyperbolic.hyp_utils import mobius_add, mobius_mm, mobius_prod, logmap0


class HypLSTMCellHyp(HypRecurrentCell):
    def __init__(self, input_size, hidden_size, dropout, activation):
        super().__init__(input_size, hidden_size, 4 * hidden_size, dropout, activation)

    def recurrent_step(self, xi, state):
        if not isinstance(state, tuple):
            state = (state, state)
        hx, cx = state

        drop_Wx = self.dropout(self.Wx, self.training)
        drop_Wh = self.dropout(self.Wh, self.training)

        gates = mobius_add(
            mobius_add(mobius_mm(xi, drop_Wx), self.bx),
            mobius_add(mobius_mm(hx, drop_Wh), self.bh)
        )

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, -1)

        input_gate = torch.sigmoid(logmap0(input_gate))
        forget_gate = torch.sigmoid(logmap0(forget_gate))
        cell_gate = self.activation(cell_gate) if self.activation is not None else cell_gate
        output_gate = torch.sigmoid(logmap0(output_gate))

        cy = mobius_add(
            mobius_prod(cx, forget_gate),
            mobius_prod(input_gate, cell_gate)
        )
        cy = self.activation(cy) if self.activation is not None else cy
        hy = mobius_prod(output_gate, cy)

        return hy, cy

    def forward(self, x, hidden=None):
        return super(HypLSTMCellHyp, self).forward(x, hidden)


class HypLSTM(HypRecurrent):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=.0, activation="tanh",
                 gradual=True):
        super(HypLSTM, self).__init__(input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual,
                                      HypLSTMCellHyp)

    def forward(self, X, hidden_states=None):
        return super(HypLSTM, self).forward(X, hidden_states)
