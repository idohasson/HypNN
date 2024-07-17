import torch


class SNN(torch.nn.Module):
    def __init__(self, model):
        super(SNN, self).__init__()
        self.model = model
        self.dist_func = lambda x1, x2: torch.abs(x1 - x2).sum(-1)

    def forward(self, X):
        Y = self.model(X)
        return self.dist_func(Y[::2], Y[1::2])
