import torch
import torch.nn as nn

from model.layers.hyperbolic.activation import HypReLU, HypSigmoid, HypTanh, HypDropout
from model.layers.hyperbolic.hyp_utils import mobius_add, expmap0, mobius_mm, projection
from model.layers.utils import layer_size


class HypLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, dropout=0.):
        super(HypLinear, self).__init__()
        k = 1 / input_size ** .5
        self.weight = nn.Parameter(projection(expmap0(torch.empty(input_size, output_size).uniform_(-k, k))))
        self.bias = nn.Parameter(torch.empty(1, output_size).zero_(), requires_grad=bias)
        self.dropout = HypDropout(dropout)
        self.use_bias = bias

    def forward(self, x):
        drop_weight = self.dropout(self.weight, self.training)
        hy = mobius_mm(x, drop_weight)
        if self.use_bias:
            hy = mobius_add(hy, self.bias)
        return hy


class HypMLP(torch.nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, bias=True, dropout=0., activation=None):
        super(HypMLP, self).__init__()
        self.num_layers = num_layers
        self.layer_list = nn.ModuleList(
            [HypLinear(in_size, out_size, bias, dropout) for in_size, out_size
             in layer_size(input_size, output_size, num_layers)]
        )

        if activation == 'tanh':
            self.activation = HypTanh()
        elif activation == 'relu':
            self.activation = HypReLU()
        elif activation == 'sigmoid':
            self.activation = HypSigmoid()
        else:
            self.activation = None

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layer_list[i](x)
            if self.activation is not None and i < self.num_layers - 1:
                x = self.activation(x)
        return x
