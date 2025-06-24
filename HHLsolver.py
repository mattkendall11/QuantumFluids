from hhl_funcs.hhl import HHL
# from src.vlasov import get_L_and_psi
from examp import get_L_and_psi
from src.classical_functions import solve_linear_system
import numpy as np

from qiskit.primitives import Estimator, Sampler


# # create the matrix
# A = np.random.rand(8,8)
# A = A+A.T
#
# # create the right hand side
# b = np.random.rand(8)
L_op, psi = get_L_and_psi()
n = L_op.shape[1]
I = np.eye(n, dtype='float64')
L_dense = np.column_stack([L_op @ I[:, i] for i in range(n)])
L_dag = L_dense.conj().T

L_herm = np.block([
    [np.zeros_like(L_dense), L_dense],
    [L_dag, np.zeros_like(L_dense)]
])

psi_pad = np.concatenate([psi, np.zeros_like(psi)])

print("psi_pad norm before normalization:", np.linalg.norm(psi_pad))
print("Any NaNs in psi_pad?", np.isnan(psi_pad).any())
print("Any Infs in psi_pad?", np.isinf(psi_pad).any())

if np.isnan(psi_pad).any() or np.isinf(psi_pad).any():
    raise ValueError("psi_pad contains NaN or Inf values.")

# Normalize psi_pad for quantum state input (norm = 1)
psi_pad = psi_pad / np.linalg.norm(psi_pad)


# create the solver and solve
hhl = HHL(estimator=Estimator(), sampler=Sampler())
solution = hhl.solve(L_herm, psi_pad)

print(solve_linear_system(L_herm,psi_pad))
print(solution.solution)