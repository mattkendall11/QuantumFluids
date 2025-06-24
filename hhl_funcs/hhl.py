# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The HHL algorithm."""

from typing import Optional, Union, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation, Isometry
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal

from qiskit.circuit.library import Isometry
from hhl_funcs.numpy_matrix import NumPyMatrix

from qiskit.quantum_info import Statevector
from qiskit.primitives import BaseEstimator, BaseSampler

from hhl_funcs.hhl_result import HHLResult


class HHL:
    """

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from quantum_linear_solvers.linear_solvers.hhl import HHL
            from quantum_linear_solvers.linear_solvers.matrices import TridiagonalToeplitz
            from quantum_linear_solvers.linear_solvers.observables import MatrixFunctional

            matrix = TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2)
            right_hand_side = [1.0, -2.1, 3.2, -4.3]
            observable = MatrixFunctional(1, 1 / 2)
            rhs = right_hand_side / np.linalg.norm(right_hand_side)

            # Initial state circuit
            num_qubits = matrix.num_state_qubits
            qc = QuantumCircuit(num_qubits)
            qc.isometry(rhs, list(range(num_qubits)), None)

            hhl = HHL()
            solution = hhl.solve(matrix, qc, observable)
            approx_result = solution.observable

    References:

        [1]: Harrow, A. W., Hassidim, A., Lloyd, S. (2009).
        Quantum algorithm for linear systems of equations.
        `Phys. Rev. Lett. 103, 15 (2009), 1–15. <https://doi.org/10.1103/PhysRevLett.103.150502>`_

        [2]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020).
        Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
        `arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`_

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        sampler: Optional[Union[BaseSampler, None]] = None,
        epsilon: float = 1e-2,
    ) -> None:
        r"""
        Args:
            estimator: Estimator used for running circuit. Default is BaseEstimator
            sampler: Sampler used for running circuit, overrides estimator. Default is None.
            epsilon: Error tolerance of the approximation to the solution, i.e. if :math:`x` is the
                exact solution and :math:`\tilde{x}` the one calculated by the algorithm, then
                :math:`||x - \tilde{x}|| \le epsilon`.
        """

        super().__init__()

        self._epsilon = epsilon
        # Tolerance for the different parts of the algorithm as per [1]
        self._epsilon_r = epsilon / 3  # conditioned rotation
        self._epsilon_s = epsilon / 3  # state preparation
        self._epsilon_a = epsilon / 6  # hamiltonian simulation

        self._scaling = None  # scaling of the solution

        self.estimator = estimator
        self.sampler = sampler

        # For now the default reciprocal implementation is exact
        self._exact_reciprocal = True
        # Set the default scaling to 1
        self.scaling = 1

    def _get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float:
        """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.

        Args:
            n_l: The number of qubits to represent the eigenvalues.
            lambda_min: the smallest eigenvalue.
            lambda_max: the largest eigenvalue.

        Returns:
            The value of the scaling factor.
        """
        formatstr = "#0" + str(n_l + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
        # floating point precision can cause problems
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep

    def construct_circuit(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        neg_vals: Optional[bool] = True,
    ) -> QuantumCircuit:
        """Construct the HHL circuit.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            neg_vals: States whether the matrix has negative eigenvalues. If False the
            computation becomes cheaper.

        Returns:
            The HHL circuit.

        Raises:
            ValueError: If the input is not in the correct format.
            ValueError: If the type of the input matrix is not supported.
        """
        # State preparation circuit - default is qiskit
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, (list, np.ndarray)):
            if isinstance(vector, list):
                vector = np.array(vector)
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb)
            # prepare the vector b in the first quantum register

            isometry = Isometry(vector / np.linalg.norm(vector), 0, 0)
            vector_circuit.append(isometry, list(range(nb)))

        # If state preparation is probabilistic the number of qubit flags should increase
        nf = 1

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
        elif isinstance(matrix, (list, np.ndarray)):
            if isinstance(matrix, list):
                matrix = np.array(matrix)

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2**vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )
            matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
        else:
            raise ValueError(f"Invalid type for matrix: {type(matrix)}.")

        # Set the tolerance for the matrix approximation
        if hasattr(matrix_circuit, "tolerance"):
            matrix_circuit.tolerance = self._epsilon_a

        # check if the matrix can calculate the condition number and store the upper bound
        if (
            hasattr(matrix_circuit, "condition_bounds")
            and matrix_circuit.condition_bounds() is not None
        ):
            kappa = matrix_circuit.condition_bounds()[1]
        else:
            kappa = 1
        # Update the number of qubits required to represent the eigenvalues
        # The +neg_vals is to register negative eigenvalues because
        # e^{-2 \pi i \lambda} = e^{2 \pi i (1 - \lambda)}
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals

        # check if the matrix can calculate bounds for the eigenvalues
        if (
            hasattr(matrix_circuit, "eigs_bounds")
            and matrix_circuit.eigs_bounds() is not None
        ):
            lambda_min, lambda_max = matrix_circuit.eigs_bounds()
            # Constant so that the minimum eigenvalue is represented exactly, since it contributes
            # the most to the solution of the system. -1 to take into account the sign qubit
            delta = self._get_delta(nl - neg_vals, lambda_min, lambda_max)
            # Update evolution time
            matrix_circuit.evolution_time = (
                2 * np.pi * delta / lambda_min / (2**neg_vals)
            )
            # Update the scaling of the solution
            self.scaling = lambda_min
        else:
            delta = 1 / (2**nl)
            print("The solution will be calculated up to a scaling factor.")

        if self._exact_reciprocal:
            reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)
            # Update number of ancilla qubits
            na = matrix_circuit.num_ancillas
        else:
            # Calculate breakpoints for the reciprocal approximation
            num_values = 2**nl
            constant = delta
            a = int(round(num_values ** (2 / 3)))

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(
                nb,
                int(
                    np.log(
                        1
                        + (
                            16.23
                            * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)
                            * kappa
                            * (2 * kappa - self._epsilon_r)
                        )
                        / self._epsilon_r
                    )
                ),
            )
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            breakpoints = []
            for i in range(0, num_intervals):
                # Add the breakpoint to the list
                breakpoints.append(a * (5**i))

                # Define the right breakpoint of the interval
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)

            reciprocal_circuit = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), degree, breakpoints, nl
            )
            na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)

        # Initialise the quantum registers
        qb = QuantumRegister(nb)  # right hand side and solution
        ql = QuantumRegister(nl)  # eigenvalue evaluation qubits
        if na > 0:
            qa = AncillaRegister(na)  # ancilla qubits
        qf = QuantumRegister(nf)  # flag qubits

        if na > 0:
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        qc.append(vector_circuit, qb[:])
        # QPE
        phase_estimation = PhaseEstimation(nl, matrix_circuit)
        if na > 0:
            qc.append(
                phase_estimation, ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas]
            )
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        # Conditioned rotation
        if self._exact_reciprocal:
            qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(
                reciprocal_circuit.to_instruction(),
                ql[:] + [qf[0]] + qa[: reciprocal_circuit.num_ancillas],
            )
        # QPE inverse
        if na > 0:
            qc.append(
                phase_estimation.inverse(),
                ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas],
            )
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        return qc

    @staticmethod
    def get_solution_vector(solution, A, b) -> np.ndarray:
        """Extracts and normalizes simulated state vector
        from LinearSolverResult.

        Args:
            solution (): HHL result instance
            A (np.ndarray): LS matrix
            b (np.ndarray): LS rhs

        Returns:
            np.ndarray: solution of the LS
        """
        nqubits = int(np.log2(len(solution.state)))
        start_idx = 2 ** (nqubits - 1)
        end_idx = start_idx + 2 ** int(np.log2(len(b)))
        solution_vector = Statevector(solution.state).data[start_idx:end_idx].real
        tmp_vec = A @ solution_vector
        norm = np.linalg.norm(tmp_vec) / np.linalg.norm(b)
        return solution_vector / norm

    @staticmethod
    def get_state_statevector(circuit: QuantumCircuit):
        """Extract the statevector of a circuit using statevector simulators

        Args:
            circuit (QuantumCircuit): the circuit
        """
        return np.real(Statevector(circuit).data)

    @staticmethod
    def get_state_qst(circuit: QuantumCircuit):
        """Extract the statevector using quantum state tomography

        Args:
            circuit (QuantumCircuit): the circuit
        """
        raise NotImplementedError("get_state_qst not implemented yet")

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
    ) -> HHLResult:
        """Solve the linear system

        Args:
            matrix (Union[List, np.ndarray, QuantumCircuit]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of the linear system

        Returns:
            HHLResult: Dataclass with solution vector of the linear system
        """

        solution = HHLResult

        solution.circuit = self.construct_circuit(matrix, vector)
        solution.state = self.get_state_statevector(solution.circuit)
        solution.solution = self.get_solution_vector(solution, matrix, vector)

        return solution
