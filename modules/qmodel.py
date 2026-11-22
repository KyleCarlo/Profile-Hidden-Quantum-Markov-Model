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

def compress_phmm_kraus(
    K_phmm,
    n,
    W,
    device="cpu",
    seed=2
):
    """
    Create a Plan7-informed HQMM with W Kraus
    operators per observation.

    K_phmm:
        Original PHMM-derived Kraus dictionary
        indexed by (j, symbol), with j=0,...,n-1.

    Returns:
        A valid complex-valued Kraus dictionary
        indexed by (w, symbol), with w=0,...,W-1.

    For W < n, the original PHMM probabilities and
    structural zeros are not guaranteed to be preserved.
    """

    if not 1 <= W <= n:
        raise ValueError("W must satisfy 1 <= W <= n")

    # numerical precision
    dtype = torch.complex128
    real_dtype = torch.float64

    # extracts the unique observation indices, 1-21
    symbol_ids = sorted({y for _, y in K_phmm})

    # creates an independent random number generator and initializes with seed
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Random complex mixing matrix Z = A+iB
    Z = torch.complex(
        torch.randn(
            (n, W), dtype=real_dtype,
            device=device, generator=generator
        ),
        torch.randn(
            (n, W), dtype=real_dtype,
            device=device, generator=generator
        )
    )

    # Z is not orthoganal, so we use QR decomposition
    # Orthonormal rows: Q @ Q^dagger = I_W
    Q_col, _ = torch.linalg.qr(Z, mode="reduced")
    # dagger to change the shape (less rows, more columns)
    # rows = W
    # columns = n
    Q = Q_col.conj().T

    K_new = {}

    # Mix the original HMM-derived Kraus operators
    for y in symbol_ids:

        # Recover the elementwise square-root observable operator for symbol y.
        sqrt_T = torch.stack(
            [
                torch.as_tensor(
                    K_phmm[(j, y)][:, j],
                    dtype=dtype,
                    device=device
                )
                for j in range(n)
            ],
            dim=1
        )

        # Generate W compressed Kraus operators by scaling each column of sqrt_T
        # using the corresponding entries in row j of the mixing matrix Q.
        #
        # Kbar_(j,y) = sqrt_T_y @ diag(Q[j, :])
        #
        # Note: The compressed operators still require Kraus
        # completeness normalization afterward.
        for j in range(W):
            K_new[(j, y)] = (
                sqrt_T * Q[j, :].unsqueeze(0)
            )

    # Restore Kraus completeness
    S = torch.zeros(
        (n, n), dtype=dtype, device=device
    )

    for A in K_new.values():
        S += A.conj().T @ A

    S = (S + S.conj().T) / 2

    eigvals, eigvecs = torch.linalg.eigh(S)

    if eigvals.min().item() <= 1e-12:
        raise ValueError(
            "Compression produced a singular normalization "
            "matrix. Try another seed or a larger W."
        )

    S_inv_sqrt = (
        eigvecs
        @ torch.diag(eigvals.rsqrt()).to(dtype)
        @ eigvecs.conj().T
    )

    K_new = {
        key: (A @ S_inv_sqrt)
            .to(torch.complex64)
            .detach()
        for key, A in K_new.items()
    }

    return K_new

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