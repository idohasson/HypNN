import random

import torch
from Levenshtein import distance
from torch.nn.functional import one_hot
from itertools import chain, repeat

def pair_loader(sequences, batch_size, seed=42):
    random.seed(seed)
    alphabet = ''.join(sorted(set.union(*map(set, sequences))))
    max_len = len(max(sequences, key=len))
    batch_size *= 2
    while True:
        rand_sequences = random.sample(sequences, batch_size)
        distances = torch.DoubleTensor([distance(*pair) for pair in zip(rand_sequences[::2], rand_sequences[1::2])])
        encoding = one_hot(
            torch.tensor(([[i for i in chain(map(alphabet.index, s), repeat(len(alphabet), max_len - len(s)))] for s in rand_sequences])),
            len(alphabet)+1).double()
        yield encoding, distances

