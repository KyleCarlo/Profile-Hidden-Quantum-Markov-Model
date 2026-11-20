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

def initialize_random_hqmm(n, s, w, device="cpu"):
    """
    Random complex HQMM initialization for arbitrary w.

    Parameters
    ----------
    n : int
        Hidden-state dimension.

    s : int
        Number of modeled observable symbols,
        excluding the start symbol (^).

    w : int
        Number of Kraus operators per symbol.

    device : str
        Device for tensor allocation.

    Returns
    -------
    p : torch.Tensor
        Random Hermitian PSD density matrix.

    K : dict
        Kraus operators indexed by (w, symbol).

        Symbol indices run from 1 to s.
        Index 0 is reserved for ^ and excluded.
    """

    # Random complex matrix
    shape = (n * s * w, n)

    Z = (
        torch.randn(shape, device=device)
        + 1j * torch.randn(shape, device=device)
    ).to(torch.complex64)

    # Orthonormalize columns (Stiefel manifold)
    kappa, _ = torch.linalg.qr(Z, mode="reduced")

    # Partition into n x n Kraus operators
    K = {}

    for y in range(s):
        symbol_idx = y + 1  # Skip ^ (index 0)

        for j in range(w):
            index = y * w + j

            K[(j, symbol_idx)] = kappa[
                index * n:(index + 1) * n, :
            ].clone()

    # Random Hermitian PSD initial density matrix
    G = (
        torch.randn((n, n), device=device)
        + 1j * torch.randn((n, n), device=device)
    ).to(torch.complex64)

    p = G @ G.conj().T
    p = p / torch.trace(p).real

    return p, K

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