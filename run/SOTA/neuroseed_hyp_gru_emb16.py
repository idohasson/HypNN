import torch
from torch import nn

from data.pair_loader import pair_loader
from data.read_data import read_nt_sequences
from model.layers.hyperbolic.hyp_gru import HypGRU
from model.layers.hyperbolic.hyp_linear import HypLinear
from model.layers.hyperbolic.hyp_snn import HypSNN
from run.train import train


class NeuroSEED_HypGRU(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, num_layers):
        super(NeuroSEED_HypGRU, self).__init__()
        self.sequence_encoder = HypGRU(input_size, hidden_size, num_layers)
        self.readout = HypLinear(hidden_size, embedding_size)

    def forward(self, sequence):
        _, enc_sequence = self.sequence_encoder(sequence)
        enc_sequence = self.readout(enc_sequence)
        return enc_sequence


torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
nt = read_nt_sequences("../../data/beta_mixcr")
train_loader = pair_loader(nt, 256, seed=42)
test_loader = pair_loader(nt, 1000, seed=43)
neuroseed_hyp_gru_emb16 = NeuroSEED_HypGRU(5, 128, 128, 1)
model = HypSNN(neuroseed_hyp_gru_emb16)

train(model, train_loader, test_loader, model_name='neuroseed_hyp_gru_emb128')
