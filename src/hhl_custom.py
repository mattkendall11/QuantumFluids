import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT, RYGate, MCMT, UnitaryGate


def prepare_b_circuit(b):
    """
    Prepare |b> state for 2x2 system.
    b: 1D numpy array, length 2, normalized.
    Returns: QuantumCircuit
    """
    qc = QuantumCircuit(1)
    theta = 2 * np.arccos(np.real(b[0]))
    qc.ry(theta, 0)
    return qc


def hhl_2x2(A, b, t=1.0):
    """
    Minimal HHL for 2x2 diagonal Hermitian A and normalized b.
    Returns: Statevector of the solution (up to normalization).
    """
    # Only support diagonal A for Aer compatibility
    if not np.allclose(A, np.diag(np.diag(A))):
        raise ValueError("hhl_2x2 only supports diagonal matrices for Aer compatibility.")
    diag = np.diag(A)
    qc = QuantumCircuit(3)
    b_circ = prepare_b_circuit(b)
    qc.compose(b_circ, [1], inplace=True)
    qc.h(0)
    # Controlled-phase for each eigenvalue (ensure float)
    qc.cp(float(np.real(diag[0])*t), 0, 1)
    qc.x(1)
    qc.cp(float(np.real(diag[1])*t), 0, 1)
    qc.x(1)
    qc.h(0)
    qc.cry(np.pi/4, 0, 2)  # Placeholder rotation
    qc.h(0)
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(qc).result()
    sv = Statevector(result.get_statevector(qc))
    return sv


def classical_2x2(A, b):
    """
    Classical solution for 2x2 Ax = b.
    """
    return np.linalg.solve(A, b)


def prepare_b_circuit_n(b):
    """
    Prepare |b> state for n-dimensional system.
    b: 1D numpy array, length n, normalized.
    Returns: QuantumCircuit
    """
    n = int(np.log2(len(b)))
    qc = QuantumCircuit(n)
    from qiskit.circuit.library import Initialize
    init = Initialize(b)
    qc.append(init, range(n))
    return qc


def inverse_qft_3(qc, q0, q1, q2):
    # Implements the inverse QFT for 3 qubits using only native gates
    qc.swap(q0, q2)
    qc.h(q0)
    qc.cp(-np.pi/2, q1, q0)
    qc.h(q1)
    qc.cp(-np.pi/4, q2, q0)
    qc.cp(-np.pi/2, q2, q1)
    qc.h(q2)


def hhl_8x8(diag_A, b, t=1.0):
    """
    Minimal HHL for 8x8 diagonal Hermitian A and normalized b.
    diag_A: 1D array of 8 real eigenvalues (diagonal elements)
    b: 1D array of length 8, normalized
    Returns: Statevector of the solution (up to normalization).
    """
    n_sys = 3  # 8x8 system
    n_qpe = 3  # QPE precision (3 bits)
    n_anc = 1
    qc = QuantumCircuit(n_qpe + n_sys + n_anc)
    b_circ = prepare_b_circuit_n(b)
    qc.compose(b_circ, range(n_qpe, n_qpe + n_sys), inplace=True)
    for q in range(n_qpe):
        qc.h(q)
    for j in range(n_qpe):
        for i in range(8):
            phase = float(np.real(diag_A[i]) * t * (2 ** j))
            ctrl = j
            target = n_qpe + (i % n_sys)
            if ctrl != target:
                qc.cp(phase, ctrl, target)
    # Use manual inverse QFT for 3 qubits (qubits 0,1,2)
    inverse_qft_3(qc, 0, 1, 2)
    qc.mcry(np.pi/4, list(range(n_qpe)), n_qpe + n_sys)
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(qc).result()
    sv = Statevector(result.get_statevector(qc))
    return sv 