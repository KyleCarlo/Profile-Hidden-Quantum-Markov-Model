import pandas as pd
import numpy as np
from modules.model import remove_silent_states
import torch

def observable_T(A, B, y):
    C = np.diag(B[y])
    return C @ A

def kraus_operators(A, B):
    n = A.shape[0]
    s = B.shape[0]
    T = {}
    for i in range(1, s):
        T_i = observable_T(A, B, i)
        for j in range(n):
            sqrt_T_i = np.zeros(T_i.shape, dtype=np.complex128)
            sqrt_T_i[:,j] += np.sqrt(T_i[:,j])
            T[j,i] = sqrt_T_i
    return T

def build_hqmm(trans_matdf, emit_matdf):
    # remove silent D states
    cols = list(trans_matdf.columns)
    d_cols = [i for i in cols if i[0] == 'D']
    trans_mat, red_cols = remove_silent_states(trans_matdf.to_numpy(), cols, d_cols)
    trans_matdf = pd.DataFrame(trans_mat, columns=red_cols, index=red_cols)

    K = kraus_operators(trans_mat, emit_matdf.to_numpy())
    return K

def convert_to_torch(K):
    K_torch = {
        key: torch.tensor(
            value,
            dtype=torch.complex64,
            requires_grad=True
        )
        for key, value in K.items()
    }
    return K_torch