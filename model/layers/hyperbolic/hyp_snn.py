import torch

from model.layers.hyperbolic.hyp_utils import hyp_distance


class HypSNN(torch.nn.Module):
    def __init__(self, model):
        super(HypSNN, self).__init__()
        self.model = model
        self.dist_func = hyp_distance

    def forward(self, X):
        Y = self.model(X)
        return self.dist_func(Y[::2], Y[1::2])
