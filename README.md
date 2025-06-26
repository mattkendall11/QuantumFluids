# Quantum Vlasov Solver Project

This project provides both classical and quantum (HHL) solvers for the 1D-1V Vlasov equation.

## Main Components

- **src/vlasov.py**:  
  construction of the Vlasov equation, initial condition, and all helper functions.  
  Provides `get_L_and_psi()` for use in quantum and classical solvers.

- **src/classical_functions.py**:  
  Classical linear system solver using NumPy.

- **hhl_funcs/**:  
  - `hhl.py`: Main HHL quantum linear system solver (Qiskit-based).
  - `numpy_matrix.py`, `linear_system_matrix.py`, `hhl_result.py`: Supporting classes for matrix representation and results.

- **HHLsolver.py**:  
  Main example script.  
  - Constructs the Vlasov operator and initial state.
  - Builds a Hermitian matrix for quantum solving.
  - Runs both the quantum HHL and classical solvers, printing results.

- **notebooks/**:  
  - `plots.ipynb`: (empty or for plotting/visualization).

- **requirements.txt**:  
  Lists required Python packages (`numpy`, `scipy`, `qiskit`, etc.).

---

## Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the quantum/classical example:**
   ```sh
   python HHLsolver.py
   ```

   This will:
   - Build the Vlasov operator and initial state.
   - Run both the quantum HHL and classical solvers.
   - Print both solutions for comparison.

---

## Structure

- `src/vlasov.py`: Core Vlasov and initial state construction.
- `src/classical_functions.py`: Classical solver.
- `hhl_funcs/`: Quantum HHL solver and supporting classes.
- `HHLsolver.py`: Main example script.
- `notebooks/`: (Optional) Jupyter notebooks for visualization.

---

## Extending

- Modify `src/vlasov.py` to change grid size, Carleman truncation, or initial conditions.
- Add plotting or analysis in `notebooks/`.

---

## Notes

- The quantum solver uses Qiskit's HHL implementation and requires the input matrix to be Hermitian and of size \(2^n \times 2^n\).
- For large grids or high truncation, the system size grows rapidly and may be infeasible to simulate classically.

