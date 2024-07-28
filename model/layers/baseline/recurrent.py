import torch
from torch import nn
from torch.autograd import Variable

from model.layers.utils import layer_size


class RecurrentCell(nn.Module):
    def __init__(self, input_size, input_hidden, hidden_hidden, dropout, activation):
        super(RecurrentCell, self).__init__()
        self.hidden_size = input_hidden
        k = 1 / input_hidden ** .5
        self.Wx = nn.Parameter(torch.empty(input_size, hidden_hidden).uniform_(-k, k))
        self.Wh = nn.Parameter(torch.empty(input_hidden, hidden_hidden).uniform_(-k, k))
        self.bx = nn.Parameter(torch.empty(1, hidden_hidden).zero_())
        self.bh = nn.Parameter(torch.empty(1, hidden_hidden).zero_())
        self.dropout = dropout
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def recurrent_step(self, xi, state):
        raise NotImplementedError

    def forward(self, x, hidden=None):
        memory = []
        for i in range(x.size(-2)):
            hidden = self.recurrent_step(x.select(-2, i), hidden)
            if isinstance(hidden, tuple):
                memory.append(hidden[0])
            else:
                memory.append(hidden)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return torch.stack(memory, dim=-2), hidden


class Recurrent(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout, activation, gradual, cell_type):
        super(Recurrent, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_list = nn.ModuleList()

        if gradual:
            for in_size, out_size in layer_size(self.input_size, self.hidden_size, self.num_layers):
                self.cell_list.append(cell_type(in_size, out_size, dropout, activation))
        else:
            self.cell_list.append(cell_type(self.input_size, self.hidden_size, dropout, activation))
            for _ in range(1, num_layers):
                self.cell_list.append(cell_type(self.hidden_size, self.hidden_size, dropout, activation))

    def forward(self, X, hidden_states=None):

        if X.dim() == 2:
            X = X.unsqueeze(0)

        hidden_forward = [cell.init_hidden(X.size(0)) for cell in self.cell_list]
        hidden_backward = [cell.init_hidden(X.size(0)) for cell in self.cell_list]

        memory_forward, memory_backward = [], []
        for xi in range(X.size(-2)):
            for layer_i in range(self.num_layers):

                hidden_state = hidden_forward[layer_i - 1] if layer_i > 0 else X.select(-2, xi)
                if isinstance(hidden_state, tuple):
                    hidden_state = hidden_state[0]

                hidden_forward[layer_i] = self.cell_list[layer_i].recurrent_step(hidden_state, hidden_forward[layer_i])

                if layer_i + 1 == self.num_layers:
                    if isinstance(hidden_forward[-1], tuple):
                        memory_forward.append(hidden_forward[-1][0])
                    else:
                        memory_forward.append(hidden_forward[-1])

                if self.bidirectional:

                    hidden_state = hidden_backward[layer_i - 1] if layer_i > 0 else X.select(-2, -(xi + 1))

                    if isinstance(hidden_state, tuple):
                        hidden_state = hidden_state[0]
                    hidden_backward[layer_i] = self.cell_list[layer_i].recurrent_step(hidden_state,
                                                                                      hidden_backward[layer_i])

                    if layer_i + 1 == self.num_layers:
                        if isinstance(hidden_backward[-1], tuple):
                            memory_backward.append(hidden_backward[-1][0])
                        else:
                            memory_backward.append(hidden_backward[-1])

        Xh = torch.stack(memory_forward, dim=-2)
        hh = hidden_forward[-1][0] if isinstance(hidden_forward[-1], tuple) else hidden_forward[-1]

        if self.bidirectional:
            Xh_back = torch.stack(memory_backward, dim=-2)
            hh_back = hidden_backward[-1][0] if isinstance(hidden_backward[-1], tuple) else hidden_backward[-1]
            Xh = torch.cat([Xh, Xh_back], dim=-1)
            hh = torch.cat([hh, hh_back], dim=-1)

        return Xh, hh
