import numpy as np
import random

def split_train_val_test(records, train_fraction=0.7, val_fraction=0.1, seed=42):
    """records: list of (header, sequence).
    Returns (train_seqs, val_seqs, test_seqs) as lists of strings.
    Test gets the remainder: 1 - train_fraction - val_fraction."""
    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1 + 1e-9:
        raise ValueError("fractions must be non-negative and sum to at most 1")
    records = list(records)                      # copy, so the original list is not reordered
    random.Random(seed).shuffle(records)
    n = len(records)
    n_train = round(train_fraction * n)
    n_train_val = round((train_fraction + val_fraction) * n)   # cumulative cut point
    train = [s for _, s in records[:n_train]]
    val   = [s for _, s in records[n_train:n_train_val]]
    test  = [s for _, s in records[n_train_val:]]
    return train, val, test

def encode_sequences(sym_idx, seqs):
    """Strings -> hmmlearn input. Wraps each sequence as '^' + seq + '$', so every path
    has to start in M0 and end in M150."""
    BEGIN = sym_idx['^']
    END = sym_idx['$']
    encoded = [[BEGIN] + [sym_idx[c] for c in s.upper()] + [END] for s in seqs]
    return np.concatenate(encoded).reshape(-1, 1), [len(e) for e in encoded]