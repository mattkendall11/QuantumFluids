# Vlasov Quantum Solver Project

This project provides:
- **vlasov_1d1v.py**: Classical mapping of the 1D-1V Vlasov equation on a small grid (e.g., 10×10). Constructs F0, F1, and nonlinear RHS for time integration.
- **quantum_solver.py**: Stubs for estimating Carleman truncation levels and building the embedded linear system for quantum algorithms (HHL, QSP). Includes `estimate_truncation_level` placeholder and notes on computational scaling.
- **run_vlasov_quantum.py**: Example script demonstrating how to set up the Vlasov system, estimate truncation order, and attempt to build the Carleman matrix (with safeguards on dimension).
- **requirements.txt**: Lists required Python packages (`numpy`, `scipy`, `qiskit`).

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run classical example:
   ```
   python run_vlasov_quantum.py
   ```
3. For actual quantum solver integration:
   - Modify `quantum_solver.py` to implement `build_carleman_matrix` for your specific problem size and truncation.
   - Use Qiskit or other quantum SDK to implement `quantum_solve_stub`.
   - Be mindful of the rapid growth in dimension: full embedding for 10×10 grid (N=100) at order ≥2 is infeasible to simulate classically.

## Notes on Truncation Level

The placeholder `estimate_truncation_level` uses a simple heuristic. For rigorous bounds, analyze the norms and Lipschitz constants of your system's F1, F2, etc., referring to quantum algorithm literature.

## Structure

- `vlasov_1d1v.py`: Core classical code; can be extended for time integration or used as part of Carleman assembly.
- `quantum_solver.py`: Contains placeholders and stubs for quantum linear solver integration.
- `run_vlasov_quantum.py`: Example orchestration script.

## Extending

- Implement adaptive timestep integration in `vlasov_1d1v.py` or integrate with `scipy.integrate.solve_ivp`.
- Expand `build_carleman_matrix` to assemble blocks for quadratic terms if N is very small or use symmetry to reduce dimension.
- Integrate Qiskit HHL or Quantum Signal Processing methods in `quantum_solver.py`.

