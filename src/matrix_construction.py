import numpy as np
from numpy.linalg import inv

# Helper: Kronecker product alias
kron = np.kron

def construct_M1(A_list):
    """
    Construct M1 = sum_{j=0}^{k-1} |j+1><j| ⊗ (A_j/(j+1))
    A_list: list of k matrices A_j (each d x d)
    Returns: M1 as a numpy array
    """
    k = len(A_list)
    d = A_list[0].shape[0]
    M1 = np.zeros((k*d, k*d), dtype=np.complex128)
    for j, A_j in enumerate(A_list[:-1]):  # Only up to k-2
        op = np.zeros((k, k))
        op[j+1, j] = 1.0
        M1 += kron(op, A_j/(j+1))
    return M1

def construct_M2(k, d):
    """
    Construct M2 = sum_{j=0}^k |0><j| ⊗ I
    Returns a rectangular matrix of shape ((k+1)*d, k*d)
    """
    M2 = np.zeros(((k+1)*d, k*d), dtype=np.complex128)
    for j in range(k):
        op = np.zeros((k+1, k))
        op[0, j] = 1.0
        M2 += kron(op, np.eye(d))
    return M2

def construct_N(M1, M2, m, p, d):
    """
    Construct N as per equation (29):
    N = sum_{i=0}^m |i+1><i| ⊗ M2 (I-M1)^{-1} + sum_{i=m+1}^{m+p-1} |i+1><i| ⊗ I
    M1: matrix from construct_M1 (k*d, k*d)
    M2: matrix from construct_M2 ((k+1)*d, k*d)
    m, p: integers
    d: block size
    Returns: N as a numpy array
    """
    N_dim = (m+p+1)*d
    N = np.zeros((N_dim, N_dim), dtype=np.complex128)
    I_M1_inv = inv(np.eye(M1.shape[0]) - M1)
    block1 = M2 @ I_M1_inv  # shape ((k+1)*d, k*d)
    # First sum: insert block1 into (i+1, i) block positions
    for i in range(m+1):
        row_start = (i+1)*d
        row_end = (i+2)*d
        col_start = i*d
        col_end = (i+1)*d
        N[row_start:row_end, col_start:col_end] = block1[:d, :d]
    # Second sum: identity blocks
    for i in range(m+1, m+p):
        row_start = (i+1)*d
        row_end = (i+2)*d
        col_start = i*d
        col_end = (i+1)*d
        N[row_start:row_end, col_start:col_end] = np.eye(d)
    return N

def construct_L(N):
    """
    Construct L = I - N
    N: matrix from construct_N
    Returns: L as a numpy array
    """
    return np.eye(N.shape[0]) - N 