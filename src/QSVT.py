import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def sum_even_odd_circ(x, phi, ancilla_wire, wires):
    """QSVT circuit implementing sum of even and odd polynomial approximations."""
    phi1, phi2 = phi[: len(phi) // 2], phi[len(phi) // 2 :]
    block_encode = qml.BlockEncode(x, wires=wires)

    qml.Hadamard(wires=ancilla_wire)  # equal superposition

    dim = x.shape[0] if x.ndim > 0 else 1
    projectors_even = [qml.PCPhase(angle, dim=dim, wires=wires) for angle in phi1]
    qml.ctrl(qml.QSVT, control=(ancilla_wire,), control_values=(0,))(block_encode, projectors_even)

    projectors_odd = [qml.PCPhase(angle, dim=dim, wires=wires) for angle in phi2]
    qml.ctrl(qml.QSVT, control=(ancilla_wire,), control_values=(0,))(block_encode, projectors_odd)

    qml.Hadamard(wires=ancilla_wire)  # un-prepare superposition


def qsvt_matrix_inversion(L, phi_vec, kappa=4, initial_phase_angles=None, max_iter=100, tol=5e-5):
    """
    Solve L y = phi_vec using QSVT matrix inversion.

    Args:
        L (np.ndarray): Hermitian matrix (n x n).
        phi_vec (np.ndarray): Vector phi (length n).
        kappa (float): Condition number for approximation domain [1/kappa, 1].
        initial_phase_angles (np.ndarray or None): Initial phase angles for optimization.
        max_iter (int): Maximum iterations for phase angle optimization.
        tol (float): Tolerance for optimization stopping criterion.

    Returns:
        np.ndarray: Solution vector y = L^{-1} phi_vec (approximate).
    """
    n = L.shape[0]

    # Scale factor s (approximate, can be tuned)
    s = 0.1

    # Define target function s*(1/x) for x in [1/kappa, 1]
    def target_func(x):
        return s / x

    # Initialize or use given phase angles
    if initial_phase_angles is None:
        np.random.seed(42)
        phase_len = 51
        phi = np.random.rand(phase_len)
    else:
        phi = initial_phase_angles

    # Domain samples for optimization
    samples_x = np.linspace(1 / kappa, 1, 100)

    # Loss function for optimizing phase angles
    def loss_func(phi_angles):
        loss = 0
        for x in samples_x:
            qsvt_mat = qml.matrix(
                sum_even_odd_circ, wire_order=["ancilla", *range(n)]
            )(x, phi_angles, ancilla_wire="ancilla", wires=list(range(n)))
            val = qsvt_mat[0, 0]
            loss += (np.real(val) - target_func(x)) ** 2
        return loss / len(samples_x)

    # Optimize phase angles with Adagrad optimizer
    opt = qml.AdagradOptimizer(stepsize=0.1)
    cost = loss_func(phi)
    iteration = 0
    while cost > tol and iteration < max_iter:
        iteration += 1
        phi, cost = opt.step_and_cost(loss_func, phi)
        if iteration % 10 == 0 or iteration == 1:
            print(f"Iteration {iteration}, loss: {cost}")

    print(f"Optimization done with final loss: {cost}")

    # Now construct a QNode to apply the QSVT polynomial approximation (inverse)
    dev = qml.device("lightning.qubit", wires=list(range(n)))

    @qml.qnode(dev)
    def qsvt_inversion_circuit(input_state):
        # Prepare |phi> state
        qml.StatePrep(input_state, wires=range(n))
        # Apply Block encoding of L
        block_encode = qml.BlockEncode(L, wires=range(n))

        # Apply QSVT with optimized phases
        projectors_even = [qml.PCPhase(angle, dim=n, wires=range(n)) for angle in phi[: len(phi) // 2]]
        projectors_odd = [qml.PCPhase(angle, dim=n, wires=range(n)) for angle in phi[len(phi) // 2 :]]

        # Sum of even and odd polynomial QSVTs with ancilla would require more wires,
        # here we do a simplified single QSVT application for demo purposes:
        qml.QSVT(block_encode, projectors_even + projectors_odd)

        # Return final state
        return qml.state()

    # Normalize input vector phi_vec to a quantum state
    norm_phi = np.linalg.norm(phi_vec)
    if norm_phi == 0:
        raise ValueError("Input vector phi_vec must be non-zero.")
    input_state = phi_vec / norm_phi

    # Run circuit and extract output quantum state corresponding to y
    output_state = qsvt_inversion_circuit(input_state)

    # The output quantum state corresponds to the vector y scaled by some factor
    # We approximate y by projecting the output state onto computational basis
    # Assuming the dimension is small, the output_state vector corresponds to amplitudes of y
    y_approx = output_state * norm_phi / s  # rescale by input norm and scale factor s

    # Return only the real part as final solution vector
    return np.real(y_approx)


# Example usage:
if __name__ == "__main__":
    # Example 2x2 Hermitian matrix L
    n = 16
    matrix = np.random.rand(n, n)
    L = matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    # L = matrix /
    # L = np.array([[0.1, 0.2], [0.2, 0.4]])
    # Example vector phi
    vector = np.random.rand(n)
    phi_vec = vector/np.linalg.norm(vector)
    # phi_vec = np.array([1.0, 0.0])

    y_solution = qsvt_matrix_inversion(L, phi_vec)

    print("Approximate solution y to L y = phi:")
    print(y_solution)
