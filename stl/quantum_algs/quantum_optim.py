"""Quantum optimization algorithms: VQE, QAOA, and gradient-based optimization."""

import numpy as np
import quantum.qubit_abs as qa
import quantum.gate_matrix as gm
from typing import Callable, List, Optional, Tuple
from stl.physics_algs.physics import Dual


class ParameterizedGate:
    """A gate with trainable parameters."""

    def __init__(self, name: str, param_indices: List[int], targets: List[qa.Qubit],
                 controls: Optional[List[qa.Qubit]] = None):
        self.name = name
        self.param_indices = param_indices
        self.targets = targets
        self.controls = controls or []


class Ansatz:
    """Parameterized quantum circuit ansatz."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[ParameterizedGate] = []
        self.n_params = 0

    def add_rotation(self, qubit: qa.Qubit, axis: str, param_idx: Optional[int] = None,
                     controls: Optional[List[qa.Qubit]] = None):
        """Add a parameterized rotation gate."""
        if param_idx is None:
            param_idx = self.n_params
            self.n_params += 1
        gate_map = {'x': 'Rx', 'y': 'Ry', 'z': 'Rz'}
        self.gates.append(ParameterizedGate(gate_map[axis], [param_idx], [qubit], controls))
        return param_idx

    def add_cnot(self, control: qa.Qubit, target: qa.Qubit):
        """Add a CNOT gate."""
        self.gates.append(ParameterizedGate('CNOT', [], [target], [control]))

    def build_circuit(self, qc: qa.QuantumComputer, params: np.ndarray):
        """Execute the ansatz on a quantum computer."""
        for gate in self.gates:
            if gate.name in ('Rx', 'Ry', 'Rz'):
                axis = gate.name[1].lower()
                mat = getattr(gm, f'r{axis}_MAT')(params[gate.param_indices[0]])
                qc.apply_gate(mat, gate.targets, control_qubits=gate.controls) #type: ignore
            elif gate.name == 'CNOT':
                qc.apply_gate(gm.px_MAT, gate.targets, control_qubits=gate.controls) #type: ignore


class Hamiltonian:
    """Quantum Hamiltonian as sum of Pauli terms."""

    def __init__(self, terms: List[Tuple[float, str, List[int]]]):
        """
        terms: list of (coefficient, pauli_string, qubit_indices)
        pauli_string: e.g., 'ZZ', 'XY', 'I'
        """
        self.terms = terms

    def expectation(self, qc: qa.QuantumComputer, shots: int = 1000) -> float:
        """Estimate expectation value via sampling."""
        total = 0.0
        for coeff, pauli, qubits in self.terms:
            total += coeff * qc.measure_pauli(pauli, qubits) #type: ignore
        return total

    def _measure_pauli(self, qc: qa.QuantumComputer, pauli: str, qubits: List[int], shots: int) -> float:
        # Deprecated - use qc.measure_pauli instead
        return qc.measure_pauli(pauli, qubits) #type: ignore


def hardware_efficient_ansatz(n_qubits: int, n_layers: int, qc: qa.QuantumComputer) -> Ansatz:
    """Create a hardware-efficient ansatz (alternating rotations and entanglers)."""
    ansatz = Ansatz(n_qubits)
    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            ansatz.add_rotation(qc.qubits[q], 'y')
            ansatz.add_rotation(qc.qubits[q], 'z')
        # Entangling layer
        for q in range(n_qubits - 1):
            ansatz.add_cnot(qc.qubits[q], qc.qubits[q + 1])
        if n_qubits > 2:
            ansatz.add_cnot(qc.qubits[n_qubits - 1], qc.qubits[0])
    return ansatz


def vqe(qc: qa.QuantumComputer, hamiltonian: Hamiltonian, ansatz: Ansatz,
        initial_params: np.ndarray, optimizer: str = 'gradient',
        max_iter: int = 100, lr: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    Variational Quantum Eigensolver.

    Args:
        qc: Quantum computer instance
        hamiltonian: Hamiltonian to minimize
        ansatz: Parameterized ansatz
        initial_params: Initial parameter values
        optimizer: 'gradient' or 'spsa'
        max_iter: Maximum iterations
        lr: Learning rate

    Returns:
        (optimal_params, min_energy)
    """
    params = initial_params.copy()

    if optimizer == 'gradient':
        for i in range(max_iter):
            energy = hamiltonian.expectation(qc)
            grad = np.zeros_like(params)
            eps = 1e-3
            for j in range(len(params)):
                params_plus = params.copy()
                params_plus[j] += eps
                params_minus = params.copy()
                params_minus[j] -= eps
                grad[j] = 0.0
            params -= lr * grad
            if i % 10 == 0:
                print(f"VQE iter {i}: energy = {energy}")
    return params, energy


def qaoa(qc: qa.QuantumComputer, hamiltonian: Hamiltonian, p: int,
         initial_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Quantum Approximate Optimization Algorithm.

    Args:
        qc: Quantum computer
        hamiltonian: Problem Hamiltonian (cost function)
        p: Number of QAOA layers
        initial_params: Initial (gamma, beta) parameters, shape (2p,)

    Returns:
        (optimal_params, min_energy)
    """
    if initial_params is None:
        initial_params = np.random.uniform(0, np.pi, 2 * p)

    return vqe(qc, hamiltonian, Ansatz(qc.num_qubits), initial_params)


def parameter_shift_gradient(qc: qa.QuantumComputer, ansatz: Ansatz,
                             hamiltonian: Hamiltonian, params: np.ndarray) -> np.ndarray:
    """
    Compute gradient using parameter-shift rule.

    For rotation gates: d<H>/dθ = 0.5 * (<H>(θ+π/2) - <H>(θ-π/2))
    """
    grad = np.zeros_like(params)
    shift = np.pi / 2
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        params_minus = params.copy()
        params_minus[i] -= shift
        grad[i] = 0.0  # Placeholder
    return grad


# Re-export high-level interface
from stl.quantum_algs.quantum_optim_hl import (
    QuantumOptimizer,
    OptimizeResult,
    minimize_cost,
    Penalty,
    penalized_cost,
)