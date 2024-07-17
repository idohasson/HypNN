import os
import pandas as pd


def read_nt_sequences(path):
    nt_sequences = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt") and file.split(".")[0][-1].isnumeric():
                file_path = os.path.join(root, file)
                nt_sequences.extend(pd.read_csv(file_path, sep='\t', usecols=['nSeqCDR3'])['nSeqCDR3'].to_list())
    filtered = filter(lambda s: 5 * 3 <= len(s) <= 20 * 3, nt_sequences)
    unique_sequences = set(filtered)
    return sorted(unique_sequences)


def read_aa_sequences(path):
    aa_sequences = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt") and file.split(".")[0][-1].isnumeric():
                file_path = os.path.join(root, file)
                aa_sequences.extend(pd.read_csv(file_path, sep='\t', usecols=['aaSeqCDR3'])['aaSeqCDR3'].to_list())
    filtered = filter(lambda s: 5 <= len(s) <= 20, aa_sequences)
    unique_sequences = set(filtered)
    return sorted(unique_sequences)
