import os
import random
from collections import Counter

import pandas as pd
from Levenshtein import distance

from data.read_data import read_nt_sequences, read_aa_sequences

nt_sequences = []
aa_sequences = []
for root, dirs, files in os.walk("../../data/beta_mixcr"):
    for file in files:
        if file.endswith(".txt") and file.split(".")[0][-1].isnumeric():
            file_path = os.path.join(root, file)
            nt_sequences.extend(pd.read_csv(file_path, sep='\t', usecols=['nSeqCDR3'])['nSeqCDR3'].to_list())
            aa_sequences.extend(pd.read_csv(file_path, sep='\t', usecols=['aaSeqCDR3'])['aaSeqCDR3'].to_list())

n_files = sum(
    1 for root, dirs, files in os.walk("../../data/beta_mixcr")
    for file in files
    if file.endswith(".txt") and file.split(".")[0][-1].isnumeric())
print("Number of repertoires", n_files)
print("NT - total sequences", len(nt_sequences))
print("AA - total sequences", len(aa_sequences))
nt_sequences = list(set(nt_sequences))
aa_sequences = list(set(aa_sequences))
print("NT - total unique sequences", len(nt_sequences))
print("AA - total unique sequences", len(aa_sequences))
nt_counter = Counter(map(len, nt_sequences))
aa_counter = Counter(map(len, aa_sequences))
total_nt = sum(nt_counter.values())
total_aa = sum(aa_counter.values())
nt_filtered_out = sum(c for l, c in nt_counter.items() if not (5 * 3 <= int(l) <= 20 * 3))
aa_filtered_out = sum(c for l, c in aa_counter.items() if not (5 <= int(l) <= 20))
print("NT - filtered out", nt_filtered_out)
print("AA - filtered out", aa_filtered_out)

nt = read_nt_sequences("../../data/beta_mixcr")

random.seed(42)
len_nt = 0
dist_nt = 0
for i in range(500):
    print("{}".format(i), end="\r", flush=True)
    rand_sequences = random.sample(nt, 256 * 2)
    len_nt += sum(map(len, rand_sequences))
    dist_nt += sum(distance(*pair) for pair in zip(rand_sequences[::2], rand_sequences[1::2]))
print("NT - lengths", len_nt / (256 * 2 * 500))
print("NT - pairwise distances", dist_nt / (256 * 500))

aa = read_aa_sequences("../../data/beta_mixcr")

random.seed(42)
len_aa = 0
dist_aa = 0
for i in range(500):
    print("{}".format(i), end="\r", flush=True)
    rand_sequences = random.sample(aa, 256 * 2)
    len_aa += sum(map(len, rand_sequences))
    dist_aa += sum(distance(*pair) for pair in zip(rand_sequences[::2], rand_sequences[1::2]))
print("AA - lengths", len_aa / (256 * 2 * 500))
print("AA - pairwise distances", dist_aa / (256 * 500))
