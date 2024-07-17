import torch
from torch import nn
from data.pair_loader import pair_loader
from data.read_data import read_nt_sequences
from run.train import train
from model.layers.hyperbolic.hyp_snn import HypSNN

class NeuroSEED_GRU(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, num_layers):
        super(NeuroSEED_GRU, self).__init__()
        self.sequence_encoder = nn.GRU(input_size, hidden_size, num_layers)
        self.readout = nn.Linear(hidden_size, embedding_size)

    def forward(self, sequence):
        sequence = sequence.transpose(0, 1)
        _, enc_sequence = self.sequence_encoder(sequence)
        enc_sequence = enc_sequence[-1]
        enc_sequence = self.readout(enc_sequence)
        return enc_sequence


torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
nt = read_nt_sequences("../../data/beta_mixcr")
train_loader = pair_loader(nt, 256, seed=42)
test_loader = pair_loader(nt, 1000, seed=43)
neuroseed_gru_emb16 = NeuroSEED_GRU(5, 128, 128, 1)
model = HypSNN(neuroseed_gru_emb16)

train(model, train_loader, test_loader, model_name='neuroseed_gru_emb128')