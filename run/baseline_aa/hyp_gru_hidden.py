import torch

from data.pair_loader import pair_loader
from data.read_data import read_aa_sequences
from model.layers.hyperbolic.hyp_gru import HypGRU
from model.layers.hyperbolic.hyp_snn import HypSNN
from run.embedding import HiddenEmbedding
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
aa = read_aa_sequences("../../data/beta_mixcr")
train_loader = pair_loader(aa, 256, seed=42)
test_loader = pair_loader(aa, 1000, seed=43)
hyp_gru = HypGRU(20 + 1, 8, 3)
hyp_gru_hidden = HiddenEmbedding(hyp_gru)
model = HypSNN(hyp_gru_hidden)

train(model, train_loader, test_loader, model_name='hyp_gru_hidden_aa')
