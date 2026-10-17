"""Quantum optimization building blocks: ansatz, Hamiltonian, expectations.

This module provides the pieces the variational loop is built from:

    Ansatz                       a parameterized circuit you can execute
    hardware_efficient_ansatz    a ready-made rotation + entangler ansatz
    Hamiltonian                  a weighted sum of Pauli terms
    measure_pauli_expectation    <psi|P|psi> for a Pauli string

The optimizer itself (gradient / SPSA / COBYLA) lives in
`stl.quantum_algs.quantum_optim_hl.QuantumOptimizer`, which drives the ansatz
with these expectations. Gate matrices come from `quantum.gate_matrix`.
"""

import numpy as np

import quantum.qubit_abs as qa
import quantum.gate_matrix as gm
from typing import List, Optional, Tuple


def pauli_string_to_matrices(pauli_str: str) -> List[np.ndarray]:
    """Convert a Pauli string into its list of 2x2 matrices."""
    pauli_map = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return [pauli_map[p.upper()] for p in pauli_str]


def pauli_string_to_tensor(pauli_str: str) -> np.ndarray:
    """Tensor product of the Pauli matrices in `pauli_str`."""
    matrices = pauli_string_to_matrices(pauli_str)
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


class Ansatz:
    """A parameterized circuit: named gates plus a pool of free parameters.

    Each rotation gate records the index of the parameter it uses, so
    `build_circuit(qc, params)` binds `params[param_idx]` to that gate. The
    same gate list can therefore be replayed with a different parameter vector
    on every optimizer step, which is what makes it an ansatz.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Tuple[str, List[int], List[int], Optional[int]]] = []  # (name, targets, controls, param_idx)
        self._n_params = 0

    def _rotation(self, name: str, qubit: int, param_idx: Optional[int]) -> int:
        if param_idx is None:
            param_idx = self._n_params
            self._n_params += 1
        self.gates.append((name, [qubit], [], param_idx))
        return param_idx

    def rx(self, qubit: int, param_idx: Optional[int] = None) -> int:
        return self._rotation("RX", qubit, param_idx)

    def ry(self, qubit: int, param_idx: Optional[int] = None) -> int:
        return self._rotation("RY", qubit, param_idx)

    def rz(self, qubit: int, param_idx: Optional[int] = None) -> int:
        return self._rotation("RZ", qubit, param_idx)

    def phase(self, qubit: int, param_idx: Optional[int] = None) -> int:
        return self._rotation("P", qubit, param_idx)

    def cnot(self, control: int, target: int):
        self.gates.append(("CNOT", [control, target], [], None))

    def cz(self, control: int, target: int):
        self.gates.append(("CZ", [control, target], [], None))

    def build_circuit(self, qc: 'qa.QuantumComputer', params: np.ndarray):
        """Execute the ansatz on `qc`, binding each rotation to its parameter."""
        rotation_matrices = {
            "RX": gm.rx_MAT,
            "RY": gm.ry_MAT,
            "RZ": gm.rz_MAT,
            "P": gm.p_MAT,
        }
        for gate_name, targets, controls, param_idx in self.gates:
            if gate_name in rotation_matrices and param_idx is not None:
                matrix = rotation_matrices[gate_name](params[param_idx])
                qc.apply_gate(matrix, targets)
            elif gate_name in ("CNOT", "CX"):
                qc.apply_gate(gm.px_MAT(), [targets[1]], control_qubits=[targets[0]])
            elif gate_name == "CZ":
                qc.apply_gate(gm.pz_MAT(), [targets[1]], control_qubits=[targets[0]])

    @property
    def n_params(self) -> int:
        return self._n_params

    @n_params.setter
    def n_params(self, value: int):
        self._n_params = value


def hardware_efficient_ansatz(n_qubits: int, n_layers: int,
                              qc: Optional['qa.QuantumComputer'] = None) -> 'Ansatz':
    """Alternating rotation and entangling layers, one RY + RZ per qubit.

    The `qc` argument is accepted for call-site symmetry but not needed: the
    ansatz is just a description until `build_circuit` is called.
    """
    ansatz = Ansatz(n_qubits)
    for _ in range(n_layers):
        for qubit in range(n_qubits):
            ansatz.ry(qubit)
            ansatz.rz(qubit)
        for qubit in range(n_qubits - 1):
            ansatz.cnot(qubit, qubit + 1)
        if n_qubits > 2:
            ansatz.cnot(n_qubits - 1, 0)
    return ansatz


class Hamiltonian:
    """A Hamiltonian expressed as a weighted sum of Pauli strings."""

    def __init__(self, terms: List[Tuple[float, str, List[int]]]):
        """
        terms: list of (coefficient, pauli_string, qubit_indices)
        pauli_string: e.g. 'ZZ', 'XY', 'I'
        """
        self.terms = terms

    def expectation(self, qc: 'qa.QuantumComputer', shots: int = 1000) -> float:
        """Estimate <H> as the weighted sum of term expectations."""
        total = 0.0
        for coefficient, pauli, qubits in self.terms:
            total += coefficient * measure_pauli_expectation(qc, pauli, qubits)
        return total


def measure_pauli_expectation(qc: 'qa.QuantumComputer', pauli_str: str,
                              qubits: List[int]) -> float:
    """Estimate <psi|P|psi> for a Pauli string on the given qubit indices.

    Delegates to `QuantumComputer.measure_pauli`, which rotates each qubit into
    the right basis, samples, and restores the state, so it can be called
    repeatedly during an optimization without disturbing the circuit.
    """
    return qc.measure_pauli(pauli_str, qubits)
