import numpy as np
from modules.sequences import encode_sequences
import matplotlib.pyplot as plt

def description_accuracy(m, sym_idx, seqs):
    """Per-sequence PHMM description accuracy."""

    da = np.empty(len(seqs))
    s = 21  # 20 amino acids + terminal '$'

    for i, seq in enumerate(seqs):
        X, _ = encode_sequences(sym_idx, [seq])

        try:
            # PHMM processes ^ + amino acids + $
            ln_p = m.score(X)
        except ValueError:
            ln_p = -np.inf

        # Exclude the deterministic initial ^ from DA length
        length = len(X) - 1

        x = 1.0 + ln_p / (length * np.log(s))
        da[i] = x if x > 0 else np.tanh(x / 8.0)

    return da