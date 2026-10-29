import numpy as np

def observable_T(A, B, y):
    C = np.diag(B[y])
    return C @ A

def kraus_operators(A, B):
    n = A.shape[0]
    s = B.shape[0]
    T = {}
    for i in range(s):
        T_i = observable_T(A, B, i)
        for j in range(n):
            sqrt_T_i = np.zeros(T_i.shape, dtype=np.complex128)
            sqrt_T_i[:,j] += np.sqrt(T_i[:,j])
            T[j,i] = sqrt_T_i
    return T