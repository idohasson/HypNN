"""
    Part of this code was adapted from "Deep Squared Euclidean
    Approximation to the Levenshtein Distance for DNA Storage"
    by 	A. J. X. Guo, C. Liang, and Q.-H. Hou.
    for more details visit https://github.com/aalennku/DSEE
"""

import torch
import torch.nn.functional as F
from torch import nn


class LinearEmbedding(torch.nn.Module):
    def __init__(self, model):
        super(LinearEmbedding, self).__init__()
        self.model = model

    def forward(self, X):
        return self.model(X.flatten(-2))


class HiddenEmbedding(torch.nn.Module):
    def __init__(self, model):
        super(HiddenEmbedding, self).__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)[1]


class UnfoldingEmbedding(torch.nn.Module):
    def __init__(self, model):
        super(UnfoldingEmbedding, self).__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)[0].flatten(-2)


class DSEE_Embedding(torch.nn.Module):
    def __init__(self, model):
        super(DSEE_Embedding, self).__init__()
        self.model = model

    def forward(self, X):
        return self.model(X.mT)


class FCEmbedding(torch.nn.Module):
    def __init__(self, model, fc):
        super(FCEmbedding, self).__init__()
        self.model = model
        self.fc = fc

    def forward(self, X):
        return self.fc(self.model(X)[0].flatten(-2))


class Twin(torch.nn.Module):
    def __init__(self, model, metric='SE', rescale=True):
        super(Twin, self).__init__()
        self.model = model
        self.rescale_dict = dict()
        self.rescale_dict['SE'] = 1 / 1.4142135623730951
        self.rescale_dict['L1'] = 1 / 1.1283791670955126
        self.rescale_dict['EU'] = 6.344349953316304
        assert metric in self.rescale_dict
        self.metric = metric
        self.rescale = rescale

    def forward(self, x):
        x, y = x[::2], x[1::2]
        # x, y = torch.unbind(x, dim=1)
        if self.rescale:
            xx = self.model(x) * self.rescale_dict[self.metric]
            yy = self.model(y) * self.rescale_dict[self.metric]
        if self.metric == 'SE':
            return torch.sum((xx - yy) ** 2, dim=-1)
        elif self.metric == 'L1':
            return torch.sum(torch.abs(xx - yy), dim=-1)
        elif self.metric == 'EU':
            return torch.linalg.norm(xx - yy, dim=-1)


class M_GRU(nn.Module):
    def __init__(self, length=160, output_dim=40, n_hidden=64):
        super(M_GRU, self).__init__()

        self.length = length
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.gru_cell = nn.GRU(input_size=5, hidden_size=n_hidden,
                               num_layers=2, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.length * self.n_hidden * 2, output_dim * 4)
        self.fc2 = nn.Linear(output_dim * 4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01)

    def forward(self, x):
        x = torch.transpose(x, -1, -2)
        x, _ = self.gru_cell(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        return x


class M_RNN(nn.Module):
    def __init__(self, length=160, output_dim=40, n_hidden=64):
        super(M_RNN, self).__init__()

        self.length = length
        self.output_dim = output_dim
        self.n_hidden = n_hidden

        self.rnn_cell = nn.RNN(input_size=5, hidden_size=n_hidden,
                               num_layers=2, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.length * self.n_hidden * 2, output_dim * 4)
        self.fc2 = nn.Linear(output_dim * 4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01)

    def forward(self, x):
        x = torch.transpose(x, -1, -2)
        x, _ = self.rnn_cell(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        return x
