import torch
from data.pair_loader import pair_loader
from data.read_data import read_nt_sequences
from model.layers.baseline.snn import SNN
from model.layers.baseline.base_mlp import MLP
from run.embedding import LinearEmbedding
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
nt = read_nt_sequences("../../data/beta_mixcr")
train_loader = pair_loader(nt, 256, seed=42)
test_loader = pair_loader(nt, 1000, seed=43)
mlp = MLP(5*60, 16, 3, activation="tanh")
mlp_flattened = LinearEmbedding(mlp)
model = SNN(mlp_flattened)

train(model, train_loader, test_loader, model_name='mlp_flattened_nt')
