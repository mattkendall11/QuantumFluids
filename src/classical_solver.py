import numpy as np
from numpy.linalg import norm, cond, solve

def generate_random_A(d, k, seed=None):
    """
    Generate a list of k random d x d matrices A_j with suitable conditioning.
    """
    rng = np.random.default_rng(seed)
    A_list = []
    for _ in range(k):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        # Improve conditioning
        U, s, Vh = np.linalg.svd(A)
        s = np.linspace(1, 10, d)  # Set singular values
        A = (U * s) @ Vh
        A_list.append(A)
    return A_list

def generate_random_psi_in(dim, seed=None):
    """
    Generate a random normalized vector psi_in with ||psi_in||_2 < 1.
    """
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(dim,)) + 1j * rng.normal(size=(dim,))
    psi /= norm(psi)
    psi *= 0.9  # Ensure norm < 1
    return psi

def solve_linear_system(L, psi_in):
    """
    Solve L|psi> = |psi_in> classically, with error handling for singular matrices.
    """
    try:
        psi = solve(L, psi_in)
        return psi
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Matrix L is singular or ill-conditioned: {e}") 