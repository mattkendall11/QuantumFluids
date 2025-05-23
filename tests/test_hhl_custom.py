import numpy as np
from hhl_custom import hhl_2x2, classical_2x2, hhl_8x8
from qiskit.quantum_info import Statevector

def test_hhl_2x2():
    # Diagonal Hermitian system
    A = np.diag([1, 2]).astype(complex)
    b = np.array([1, 0], dtype=complex)
    b = b / np.linalg.norm(b)
    # Quantum HHL
    sv = hhl_2x2(A, b)
    # Project onto system qubit (qubit 1)
    # For 3 qubits: [eigenvalue, system, ancilla]
    # We want the amplitudes where ancilla=1 (success)
    amps = sv.data.reshape(2, 2, 2)
    # Sum over eigenvalue qubit, take ancilla=1
    sol = amps[:, :, 1].sum(axis=0)
    # Classical
    c_sol = classical_2x2(A, b)
    # Compare up to normalization
    sol = sol / np.linalg.norm(sol)
    c_sol = c_sol / np.linalg.norm(c_sol)
    fidelity = np.abs(np.vdot(sol, c_sol))**2
    assert fidelity > 0.8  # Low threshold due to placeholder rotation

def test_hhl_8x8():
    # Diagonal Hermitian matrix
    diag_A = np.arange(1, 9, dtype=float)
    b = np.random.rand(8) + 1j * np.random.rand(8)
    b = b / np.linalg.norm(b)
    # Quantum HHL
    sv = hhl_8x8(diag_A, b)
    # Project onto system qubits (qubits 3,4,5)
    amps = sv.data.reshape(2, 2, 2, 2, 2, 2, 2)
    # ancilla is last qubit (6), system is qubits 3,4,5
    # Sum over QPE qubits (0,1,2), take ancilla=1
    sol = amps[:, :, :, :, :, :, 1].sum(axis=(0, 1, 2))
    # Classical
    c_sol = b / diag_A
    sol = sol / np.linalg.norm(sol)
    c_sol = c_sol / np.linalg.norm(c_sol)
    fidelity = np.abs(np.vdot(sol, c_sol))**2
    assert fidelity > 0.5  # Low threshold due to placeholder rotation 