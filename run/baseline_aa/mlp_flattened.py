import torch
from data.pair_loader import pair_loader
from data.read_data import read_aa_sequences
from model.layers.baseline.snn import SNN
from model.layers.baseline.base_mlp import MLP
from run.embedding import LinearEmbedding
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
aa = read_aa_sequences("../../data/beta_mixcr")
train_loader = pair_loader(aa, 256, seed=42)
test_loader = pair_loader(aa, 1000, seed=43)
mlp = MLP((20 + 1) * 20, 8, 3, activation="tanh")
mlp_flattened = LinearEmbedding(mlp)
model = SNN(mlp_flattened)

train(model, train_loader, test_loader, model_name='mlp_flattened_aa')