import pandas as pd
import numpy as np
from hmmlearn import hmm

def init_transition_mat(match_len):
    order = ['M', 'D', 'I']
    COLS = [order[j] + f'{i+1}' for i in range(match_len - 1) for j in range (3)]
    COLS.insert(0, 'M0')
    COLS.insert(1, 'I0')
    COLS.append(f'M{match_len}')
    
    # INITIALIZE THE PROBABILITIES
    trans_matdf = pd.DataFrame(0, columns=COLS, index=COLS, dtype=np.float64)
    
    # Starting values: based on assumptions
    T_MM, T_MI, T_MD = 0.90, 0.05, 0.05      # from an internal Match state
    T_IM, T_II       = 0.50, 0.50            # from an internal Insert state
    T_DM, T_DD       = 0.90, 0.10            # from a Delete state
    BEGIN_M, BEGIN_D, BEGIN_I = 0.45, 0.05, 0.50   # from Begin (M0)
    FLANK_II         = 0.98                  # long self-loop for I0 and the last Insert (flanks)
     
    
    # Convention: trans_matdf.loc[TO, FROM]; each COLUMN sums to 1
    for j in range(match_len):
        if j == 0:                                        # Begin (M0) and I0
            trans_matdf.loc['M1', 'M0'] = BEGIN_M
            trans_matdf.loc['D1', 'M0'] = BEGIN_D
            trans_matdf.loc['I0', 'M0'] = BEGIN_I
            trans_matdf.loc['I0', 'I0'] = FLANK_II
            trans_matdf.loc['M1', 'I0'] = 1 - FLANK_II
            continue
    
        # from Match state j
        trans_matdf.loc[f'M{j+1}', f'M{j}'] = T_MM
        trans_matdf.loc[f'I{j}',   f'M{j}'] = T_MI
        if j + 1 <= match_len - 1:                                # D{n} doesn't exist
            trans_matdf.loc[f'D{j+1}', f'M{j}'] = T_MD
    
        # from Insert state j (the last one absorbs the C-terminal flank)
        if j == match_len - 1:
            trans_matdf.loc[f'I{j}',   f'I{j}'] = FLANK_II
            trans_matdf.loc[f'M{j+1}', f'I{j}'] = 1 - FLANK_II
        else:
            trans_matdf.loc[f'I{j}',   f'I{j}'] = T_II
            trans_matdf.loc[f'M{j+1}', f'I{j}'] = T_IM
    
        # from Delete state j
        trans_matdf.loc[f'M{j+1}', f'D{j}'] = T_DM
        if j + 1 <= match_len - 1:
            trans_matdf.loc[f'D{j+1}', f'D{j}'] = T_DD
    
    # loop back to the end state
    trans_matdf.loc[f'M{match_len}', f'M{match_len}'] = 1
    
    # After the loop: renormalize COLUMNS, because states near the end lose some targets
    col_sums = trans_matdf.sum(axis=0)
    trans_matdf = trans_matdf.div(col_sums.where(col_sums > 0, 1), axis=1)
    return trans_matdf

def init_emission_mat(match_len, conc=100.0, seed=None):
    rng = np.random.default_rng(seed)
    order = ['M', 'I']
    COLS = [order[j] + f'{i+1}' for i in range(match_len - 1) for j in range(2)]
    COLS.insert(0, 'M0')
    COLS.insert(1, 'I0')
    COLS.append(f'M{match_len}')

    amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    symbols = ['^'] + amino + ['$']
    emission_matdf = pd.DataFrame(0.0, columns=COLS, index=symbols, dtype=np.float64)

    emission_matdf.loc[amino, :] = 1/20              # uniform over the 20 amino acids; '$' row stays 0

    # randomization: perturb each Match state around the uniform 1/20 (columns still sum to 1)
    for k in range(1, match_len):
        emission_matdf.loc[amino, f'M{k}'] = rng.dirichlet(np.full(20, conc / 20))

    emission_matdf['M0'] = 0.0                        # Begin: silent
    emission_matdf.loc['^', 'M0'] = 1.0
    emission_matdf[f'M{match_len}'] = 0.0             # End state: silent
    emission_matdf.loc['$', f'M{match_len}'] = 1.0    # emits only '$'

    return emission_matdf

def remove_silent_states(A, state_labels, silent_labels):
    silent_idx = [state_labels.index(s) for s in silent_labels]
    keep_idx = [i for i in range(len(state_labels)) if i not in silent_idx]
    kept_labels = [state_labels[i] for i in keep_idx]

    # slice the four corners
    A_EE = A[np.ix_(keep_idx, keep_idx)]      # real   -> real
    A_ED = A[np.ix_(keep_idx, silent_idx)]    # silent -> real   (exit)
    A_DE = A[np.ix_(silent_idx, keep_idx)]    # real   -> silent (entry)
    A_DD = A[np.ix_(silent_idx, silent_idx)]  # silent -> silent

    # fold all detours through the silent states back into A_EE
    n_silent = len(silent_idx)
    fundamental = np.linalg.inv(np.eye(n_silent) - A_DD)   # (I - A_DD)^-1
    correction = A_ED @ fundamental @ A_DE

    A_reduced = A_EE + correction
    return A_reduced, kept_labels

def build_hmmlearn_model(A0, B0, startprob, len_symbols=22, trans_strength=5.0, emit_strength=20.0):
    """Profile HMM initialised at (A0, B0), with Dirichlet priors centred on them.
 
    prior = 1 + strength * initial_probs. Entries that start at 0 get no pseudocount, so
    '$' and '^' never leak out of M150 and M0, and the topology is untouched; M150 -> M150
    gets a pseudocount, which keeps its row valid even though no transition out of M150 is
    ever observed. The priors only add pseudocounts in the M-step; they don't change the
    starting values.
    """
    m = hmm.CategoricalHMM(
        n_components=A0.shape[0],
        n_features=len_symbols,
        startprob_prior=1.0 + trans_strength * startprob,
        transmat_prior=1.0 + trans_strength * A0,
        emissionprob_prior=1.0 + emit_strength * B0,
        init_params="",            # default "ste" would overwrite the matrices below with random ones
        params="ste",              # what Baum-Welch updates
        n_iter=1,                  # one EM step per fit() call; train() handles convergence
        implementation="scaling",  # several times faster than "log" here, same result
    )
    m.startprob_ = startprob.copy()
    m.transmat_ = A0.copy()
    m.emissionprob_ = B0.copy()
    return m