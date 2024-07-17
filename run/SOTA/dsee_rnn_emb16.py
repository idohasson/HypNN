import torch
from data.pair_loader import pair_loader
from data.read_data import read_nt_sequences
from run.embedding import DSEE_Embedding, Twin, M_RNN
from run.train import train

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
nt = read_nt_sequences("../../data/beta_mixcr")
train_loader = pair_loader(nt, 256, seed=42)
test_loader = pair_loader(nt, 1000, seed=43)
dsee_rnn_emb16 = M_RNN(length=60, output_dim=16, n_hidden=64)
model = DSEE_Embedding(Twin(dsee_rnn_emb16, metric="L1", rescale=True))

train(model, train_loader, test_loader, model_name='dsee_rnn_emb16')
