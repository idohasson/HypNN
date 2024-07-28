import torch.nn as nn
import torch.nn.functional as F

from model.layers.hyperbolic.hyp_utils import expmap0, logmap0, projection


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, act):
        super(HypAct, self).__init__()
        self.act = act

    def forward(self, x):
        xa = logmap0(x)
        xa = self.act(xa)
        xa = expmap0(xa)
        xa = projection(xa)
        return xa


class HypTanh(HypAct):
    def __init__(self):
        super(HypTanh, self).__init__(nn.Tanh())


class HypReLU(HypAct):
    def __init__(self):
        super(HypReLU, self).__init__(nn.ReLU())


class HypSigmoid(HypAct):
    def __init__(self):
        super(HypSigmoid, self).__init__(nn.Sigmoid())


class HypDropout(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, p):
        super(HypDropout, self).__init__()
        self.p = p

    def forward(self, x, training):
        xt = logmap0(x)
        xt = F.dropout(xt, p=self.p, training=training)
        xt = expmap0(xt)
        xt = projection(xt)
        return xt
