import torch
from data.pair_loader import pair_loader
from data.read_data import read_aa_sequences
from model.layers.hyperbolic.hyp_snn import HypSNN
from model.layers.hyperbolic.hyp_linear import HypMLP
from run.embedding import LinearEmbedding
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
aa = read_aa_sequences("../../data/beta_mixcr")
train_loader = pair_loader(aa, 256, seed=42)
test_loader = pair_loader(aa, 1000, seed=43)
hyp_mlp = HypMLP((20 + 1) * 20, 8, 3, activation="tanh")
hyp_mlp_flattened = LinearEmbedding(hyp_mlp)
model = HypSNN(hyp_mlp_flattened)

train(model, train_loader, test_loader, model_name='hyp_mlp_flattened_aa')