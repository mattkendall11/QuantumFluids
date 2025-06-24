import numpy as np

def solve_linear_system(A: np.ndarray, b: np.ndarray, method: str = 'numpy') -> np.ndarray:
    """
    Solve the linear system Ax = b using different methods.

    Parameters:
    -----------
    A : np.ndarray
        Coefficient matrix (n x n)
    b : np.ndarray
        Right-hand side vector (n,) or (n, 1)
    method : str
        Method to use: 'numpy', 'inverse', 'lu', 'cholesky'

    Returns:
    --------
    np.ndarray
        Solution vector x
    """
    # Ensure b is a column vector
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if method == 'numpy':
        # Using numpy's built-in solver (most stable)
        x = np.linalg.solve(A, b)
    elif method == 'inverse':
        # Using matrix inverse (less stable but useful for comparison)
        A_inv = np.linalg.inv(A)
        x = A_inv @ b
    elif method == 'lu':
        # LU decomposition
        from scipy.linalg import lu_solve, lu_factor
        lu, piv = lu_factor(A)
        x = lu_solve((lu, piv), b)
    elif method == 'cholesky':
        # Cholesky decomposition (for symmetric positive definite matrices)
        from scipy.linalg import cholesky, solve_triangular
        L = cholesky(A, lower=True)
        y = solve_triangular(L, b, lower=True)
        x = solve_triangular(L.T, y, lower=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    return x.flatten() if x.shape[1] == 1 else x