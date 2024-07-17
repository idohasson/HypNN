import torch

from data.pair_loader import pair_loader
from data.read_data import read_nt_sequences
from model.layers.hyperbolic.hyp_snn import HypSNN
from model.layers.hyperbolic.hyp_gru import HypGRU
from model.layers.hyperbolic.hyp_linear import HypLinear
from model.layers.hyperbolic.activation import HypReLU
from run.embedding import FCEmbedding
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
nt = read_nt_sequences("../../data/beta_mixcr")
train_loader = pair_loader(nt, 256, seed=42)
test_loader = pair_loader(nt, 1000, seed=43)
hyp_gru = HypGRU(5, 64, 2, bidirectional=True)
hyp_fc = torch.nn.Sequential(*[
    HypLinear(60*64*2, 3*4),
    HypReLU(),
    HypLinear(3*4, 3)
])
hyp_gru_unfolding_emb3 = FCEmbedding(hyp_gru, hyp_fc)
model = HypSNN(hyp_gru_unfolding_emb3)

train(model, train_loader, test_loader, model_name='hyp_gru_unfolding_emb3')
