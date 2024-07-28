import torch

from data.pair_loader import pair_loader
from data.read_data import read_nt_sequences
from model.layers.hyperbolic.hyp_linear import HypMLP
from model.layers.hyperbolic.hyp_snn import HypSNN
from run.embedding import LinearEmbedding
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
nt = read_nt_sequences("../../data/beta_mixcr")
train_loader = pair_loader(nt, 256, seed=42)
test_loader = pair_loader(nt, 1000, seed=43)
hyp_mlp = HypMLP(5 * 60, 16, 3, activation="tanh")
hyp_mlp_flattened = LinearEmbedding(hyp_mlp)
model = HypSNN(hyp_mlp_flattened)

train(model, train_loader, test_loader, model_name='hyp_mlp_flattened_nt')
