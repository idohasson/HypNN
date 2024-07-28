import torch
import torch.nn as nn

from model.layers.utils import layer_size


class Linear(nn.Module):

    def __init__(self, input_size, output_size, bias=True, dropout=0.):
        super(Linear, self).__init__()
        k = 1 / input_size ** .5
        self.weight = nn.Parameter(torch.empty(input_size, output_size).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(1, output_size).zero_(), requires_grad=bias)
        self.dropout = dropout
        self.use_bias = bias

    def forward(self, x):
        drop_weight = torch.nn.functional.dropout(self.weight, self.dropout, self.training)
        x = torch.mm(x, drop_weight)
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, bias=True, dropout=0., activation=None):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.layer_list = nn.ModuleList(
            [Linear(in_size, out_size, bias, dropout) for in_size, out_size
             in layer_size(input_size, output_size, num_layers)]
        )
        self.dropout = dropout
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layer_list[i](x)
            if self.activation is not None and i < self.num_layers - 1:
                x = self.activation(x)
        return x
