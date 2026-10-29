from modules.evaluation import description_accuracy
from modules.sequences import encode_sequences
import pandas as pd
import numpy as np
import time
import copy

def train_phmm(m, sym_idx, train_seqs, val_seqs=None, test_seqs=None, max_iter=200, tol=1e-3, verbose=True):
    """Baum-Welch (MAP), one EM step at a time.
 
    Convergence is tracked on log-likelihood + log-prior, which EM never decreases;
    hmmlearn's built-in monitor uses the likelihood alone, which can dip when priors are
    used and would stop training early.
 
    Returns (model, history). history has one row per iteration (row 0 = untrained model)
    with the CPU time spent in fit() so far, the training log-likelihood/log-posterior and,
    if given, the mean/std DA on val_seqs and test_seqs. As in the paper, when val_seqs is
    given the returned model has the parameters from the iteration with the best
    validation DA; test DA is only recorded, never used for selection.
    """
    X, lengths = encode_sequences(sym_idx, train_seqs)
    history, cpu = [], 0.0
 
    def record(it):
        ll = m.score(X, lengths)
        row = {"iter": it, "cpu_time": cpu, "loglik": ll}
        for name, seqs in (("val", val_seqs), ("test", test_seqs)):
            if seqs:
                da = description_accuracy(m, sym_idx, seqs)
                row[f"{name}_da_mean"], row[f"{name}_da_std"] = da.mean(), da.std()
        history.append(row)
        if verbose:
            extra = "".join(f"  {k} {v:.4f}" for k, v in row.items() if k.endswith("da_mean"))
            print(f"iter {it:3d}  cpu {cpu:7.1f}s  log-lik {ll:.2f}{extra}")
        return row
 
    def params():
        return copy.deepcopy((m.startprob_, m.transmat_, m.emissionprob_))
 
    best = (record(0).get("val_da_mean", -np.inf), params())
    for it in range(1, max_iter + 1):
        t0 = time.process_time()
        m.fit(X, lengths)
        cpu += time.process_time() - t0
        row = record(it)
        if val_seqs and row["val_da_mean"] > best[0]:
            best = (row["val_da_mean"], params())
        if abs(row["loglik"] - history[-2]["loglik"]) < tol:
            break
    if val_seqs:
        m.startprob_, m.transmat_, m.emissionprob_ = best[1]
    return m, pd.DataFrame(history)