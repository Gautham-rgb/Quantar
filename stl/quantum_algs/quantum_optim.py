"""Quantum optimization algorithms: VQE, QAOA, and gradient-based optimization."""

import numpy as np
import quantum.qubit_abs as qa
import quantum.gate_matrix as gm
from typing import Callable, List, Optional, Tuple, Union


def rx_gate(angle: float) -> np.ndarray:
    """RX rotation gate matrix."""
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def ry_MAT(angle: float) -> np.ndarray:
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz_MAT(angle: float) -> np.ndarray:
    c_minus = np.cos(angle / 2) - 1j * np.sin(angle / 2)
    c_plus = np.cos(angle / 2) + 1j * np.sin(angle / 2)
    return np.array([[np.cos(angle/2) - 1j*np.sin(angle/2), 0.0], [0.0, np.cos(angle/2) + 1j*np.sin(angle/2)]], dtype=complex)

def p_MAT(angle: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)

def rx(angle: float) -> np.ndarray:
    """RX rotation gate matrix."""
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def ry(angle: float) -> np.ndarray:
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(angle: float) -> np.ndarray:
    c_minus = np.cos(angle / 2) - 1j * np.sin(angle / 2)
    c_plus = np.cos(angle / 2) + 1j * np.sin(angle / 2)
    return np.array([[np.cos(angle/2) - 1j*np.sin(angle/2), 0.0], [0.0, np.cos(angle/2) + 1j*np.sin(angle/2)]], dtype=complex)

def p_gate(angle: float) -> np.ndarray:
    """Phase gate: |0><0| + e^(i*angle)|1><1|"""
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)


class ParamGate:
    """A parameterized gate that can be applied with different parameter values."""
    
    def __init__(self, name: str, gate_func: Callable, num_params: int, 
                 num_qubits: int, num_controls: int = 0):
        self.name = name
        self.gate_func = gate_func
        self.num_params = num_params
        self.num_qubits = num_qubits
        self.num_controls = num_controls
    
    def get_matrix(self, *params) -> np.ndarray:
        """Get the matrix for this gate with given parameters."""
        return self.gate_func(*params)
    
    def __call__(self, *params) -> np.ndarray:
        return self.gate_func(*params)


RX = "RX"
RY = "RY"
RZ = "RZ"
PHASE = "P"

GATE_FACTORIES = {
    "RX": (rx, 1, 1, 0),
    "RY": (ry, 1, 1, 0),
    "RZ": (rz, 1, 1, 0),
    "PHASE": (p_MAT, 1, 1, 0),
    "PHASE": (p_MAT, 1, 1, 0),
}

def pauli_string_to_matrices(pauli_str: str) -> List[np.ndarray]:
    """Convert Pauli string to list of matrices."""
    pauli_map = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return [pauli_map[p] for p in pauli_str]

def pauli_string_to_tensor(pauli_str: str) -> np.ndarray:
    """Compute tensor product of Pauli matrices."""
    mats = pauli_string_to_matrices(pauli_str)
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def pauli_expectation(qc: 'qa.QuantumComputer', pauli_str: str, qubits: List[int]) -> float:
    """
    Compute expectation value of a Pauli string on given qubits.
    
    Args:
        qc: QuantumComputer instance
        pauli_str: Pauli string like 'XX', 'YZ', 'IZZ', etc.
        qubits: List of qubit indices
    
    Returns:
        Expectation value <psi|Pauli|psi>
    """
    pauli_str = pauli_str.upper()
    if len(pauli_str) != len(qubits):
        raise ValueError("Pauli string length must match number of qubits")
    
    # Apply basis rotations to measure in Pauli basis
    qc = qc  # reference to the same qc
    qubits_list = [qc.qubits[i] for i in qubits]
    
    # Apply basis rotations before measurement
    for i, p in enumerate(zip(pauli_str.upper())):
        p = p[0]
        q = qubits[i]
        if p == 'X':
            qc.apply_gate(gm.h_MAT(), [i])
        elif p == 'Y':
            # Y basis = H * S†
            from quantum import gate_matrix as gm
            qc.apply_gate(gm.sdg_MAT(), [qubits[i]])
            qc.apply_gate(gm.h_MAT(), [i])
        elif p == 'Z':
            pass  # Z is computational basis
        elif p == 'I':
            pass
        else:
            raise ValueError(f"Unknown Pauli operator: {p}")
    
    # Measure in computational basis
    probs = np.abs(qc.state_vector.flatten()) ** 2
    probs /= probs.sum()
    
    # Compute expectation value
    # For Pauli Z, eigenvalue is +1 for |0>, -1 for |1>
    exp_val = 0.0
    for i, prob in enumerate(np.abs(qc.state_vector.flatten())**2):
        parity = bin(i).count('1') % 2
        exp_val += prob * (1 if parity == 0 else -1)
    
    return float(np.real(exp_val))


class Ansatz:
    """Parameterized quantum circuit ansatz."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Tuple[str, List[int], List[int], Optional[int]]] = []  # (gate_name, targets, controls, param_idx)
        self._n_params = 0
    
    def rx(self, qubit: int, param_idx: Optional[int] = None) -> int:
        if param_idx is None:
            param_idx = self._n_params
            self._n_params += 1
        self.gates.append(("RX", [qubit], [], param_idx))
        return param_idx
    
    def ry(self, qubit: int, param_idx: Optional[int] = None) -> int:
        if param_idx is None:
            param_idx = self._n_params
            self._n_params += 1
        self.gates.append(("RY", [qubit], [], param_idx))
        return param_idx
    
    def rz(self, qubit: int, param_idx: Optional[int] = None) -> int:
        if param_idx is None:
            param_idx = self._n_params
            self._n_params += 1
        self.gates.append(("RZ", [qubit], [], param_idx))
        return param_idx
    
    def phase(self, qubit: int, param_idx: Optional[int] = None) -> int:
        if param_idx is None:
            param_idx = self._n_params
            self._n_params += 1
        self.gates.append(("P", [qubit], [], param_idx))
        return param_idx
    
    def cnot(self, control: int, target: int):
        self.gates.append(("CNOT", [control, target], [], None))
    
    def cz(self, control: int, target: int):
        self.gates.append(("CZ", [control, target], [], None))
    
    def build_circuit(self, qc: 'qa.QuantumComputer', params: np.ndarray, gate_matrix_module=gm):
        """Execute the ansatz on a quantum computer with given parameters."""
        from quantum import gate_matrix as gm
        for gate in self.gates:
            gate_name, targets, controls, param_idx = gate
            if gate_name in ("RX", "RY", "RZ", "P") and param_idx is not None:
                param_val = params[param_idx]
                if gate_name == "RX":
                    qc.apply_gate(gm.rx_MAT(param_val), targets) #type: ignore
                elif gate_name == "RY":
                    qc.apply_gate(gm.ry_MAT(param_val), targets) #type: ignore
                elif gate_name == "RZ":
                    qc.apply_gate(gm.rz_MAT(param_val), targets) #type: ignore
                elif gate_name == "P":
                    qc.apply_gate(p_MAT(param_val), targets) #type: ignore
            elif gate_name == "CNOT" or gate_name == "CX":
                qc.apply_gate(gm.px_MAT(), [targets[1]], control_qubits=[targets[0]])
            elif gate_name == "CZ":
                qc.apply_gate(gm.pz_MAT(), [targets[1]], control_qubits=[targets[0]])
    
    @property
    def n_params(self) -> int:
        return self._n_params

    @n_params.setter
    def n_params(self, value: int):
        self._n_params = value


def hardware_efficient_ansatz(n_qubits: int, n_layers: int, qc: Optional['qa.QuantumComputer'] = None) -> 'Ansatz':
    """Create a hardware-efficient ansatz (alternating rotations and entanglers)."""
    ansatz = Ansatz(n_qubits)
    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            ansatz.ry(q)
            ansatz.rz(q)
        # Entangling layer
        for q in range(n_qubits - 1):
            ansatz.cnot(q, q + 1)
        if n_qubits > 2:
            ansatz.cnot(n_qubits - 1, 0)
    return ansatz


class Hamiltonian:
    """Quantum Hamiltonian as sum of Pauli terms."""
    
    def __init__(self, terms: List[Tuple[float, str, List[int]]]):
        """
        terms: list of (coefficient, pauli_string, qubit_indices)
        pauli_string: e.g., 'ZZ', 'XY', 'I'
        """
        self.terms = terms
    
    def expectation(self, qc: 'qa.QuantumComputer', shots: int = 1000) -> float:
        """Estimate expectation value via sampling."""
        total = 0.0
        for coeff, pauli, qubits in self.terms:
            total += coeff * measure_pauli_expectation(qc, pauli, qubits)
        return total


def measure_pauli_expectation(qc: 'qa.QuantumComputer', pauli_str: str, qubits: List[int]) -> float:
    """
    Compute expectation value <psi|Pauli|psi> for a Pauli string.
    
    Args:
        qc: QuantumComputer instance
        pauli_str: Pauli string like 'XX', 'YZ', 'IZZ', etc.
        qubits: List of qubit indices
    
    Returns:
        Expectation value <psi|Pauli|psi>
    """
    pauli_str = pauli_str.upper()
    if len(pauli_str) != len(qubits):
        raise ValueError(f"Pauli string length ({len(pauli_str)}) must match number of qubits ({len(qubits)})")
    
    # Get qubit objects
    from quantum.qubit_abs import QuantumComputer
    qc_ref = QuantumComputer(1)  # dummy for method access
    
    # We'll apply basis rotations, measure, and compute expectation
    # This is a simplified version - in practice you'd want to use the actual qc
    pauli_str = pauli_str.upper()
    targets = list(range(len(pauli_str)))  # qubit indices in the Pauli string
    
    # Apply basis rotations
    for i, p in enumerate(pauli_str):
        if pauli_str[i] == 'X':
            # X basis: H
            pass  # We'll handle this in the actual qc.apply
        elif pauli_str[i] == 'Y':
            # Y basis: S† then H
            pass
        elif pauli_str[i] == 'Z':
            pass  # Z basis is computational basis
        elif pauli_str[i] == 'I':
            pass
        else:
            raise ValueError(f"Unknown Pauli operator: {p}")
    
    # For now, we'll use a simpler approach - measure in computational basis
    # and compute expectation from probabilities
    # This is a placeholder - full implementation would rotate basis
    probs = np.abs(qc.state_vector.flatten()) ** 2
    probs = probs / probs.sum()
    
    # Compute expectation from probabilities
    # For Pauli Z: eigenvalue +1 for |0>, -1 for |1>
    # For Pauli X: +1 for |+>, -1 for |->
    # For Pauli Y: +1 for |+i>, -1 for |-i>
    # We need to rotate basis, measure, then compute expectation
    
    # Simplified: just compute for Z basis for now
    if pauli_str == 'Z' * len(qubits):
        probs = np.abs(qc.state_vector.flatten()) ** 2
        probs = probs / probs.sum()
        exp_val = 0.0
        for i, prob in enumerate(probs):
            parity = bin(i).count('1') % 2
            exp_val += prob * (1 if parity == 0 else -1)
        return float(exp_val)
    
    # For other Paulis, we'd need basis rotation
    # This is a simplified placeholder
    return 0.0


class VQE:
    """Variational Quantum Eigensolver."""
    
    def __init__(self, n_qubits: int, hamiltonian: 'Hamiltonian', 
                 ansatz_layers: int = 2, optimizer: str = 'gradient'):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.ansatz = hardware_efficient_ansatz(n_qubits=2, n_layers=ansatz_layers)  # placeholder
        self.n_params = None
        self.n_layers = ansatz_layers
        
    def cost_function(self, params: np.ndarray, qc: 'qa.QuantumComputer', 
                      hamiltonian: 'Hamiltonian') -> float:
        """Cost function for VQE: expectation value of Hamiltonian."""
        # Build ansatz with params
        ansatz = hardware_efficient_ansatz(self.n_qubits, self.n_layers)
        qc = qa.QuantumComputer(self.n_qubits)
        ansatz.build_circuit(qc, params)
        
        # Compute expectation value
        return self.hamiltonian.expectation(qc)
    
    def minimize(self, initial_params: Optional[np.ndarray] = None,
                 optimizer: str = 'gradient', max_iter: int = 100,
                 lr: float = 0.1, callback=None):
        """Run VQE optimization."""
        from stl.quantum_algs.quantum_optim_hl import QuantumOptimizer
        
        optimizer_obj = QuantumOptimizer(n_qubits=2, n_layers=2)
        
        def cost_fn(params, qc):
            ansatz = hardware_efficient_ansatz(2, 1)
            ansatz.build_circuit(qc, params)
            return 0.0  # placeholder
        
        optimizer_obj = QuantumOptimizer(n_qubits=2, n_layers=2)
        # This is a placeholder - the actual VQE would need proper implementation
        pass


def vqe(hamiltonian, ansatz, initial_params, optimizer='gradient', max_iter=100, lr=0.01):
    """
    Variational Quantum Eigensolver.
    
    Args:
        hamiltonian: Hamiltonian object
        ansatz: Ansatz object
        initial_params: Initial parameter values
        optimizer: 'gradient' or 'spsa'
        max_iter: Maximum iterations
        lr: Learning rate
    
    Returns:
        (optimal_params, min_energy)
    """
    pass


def qaoa(hamiltonian, p: int, initial_params: Optional[np.ndarray] = None) -> tuple[np.ndarray, float] | None:
    """Quantum Approximate Optimization Algorithm."""
    pass


def parameter_shift_gradient(qc, ansatz, hamiltonian, params: np.ndarray) -> np.ndarray:
    """
    Compute gradient using parameter-shift rule.
    
    For rotation gates: d<H>/dθ = 0.5 * (<H>(θ+π/2) - <H>(θ-π/2))
    """
    grad = np.zeros_like(params)
    shift = np.pi / 2
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += np.pi / 2
        params_minus = params.copy()
        params_minus[i] -= shift
        # This would need a proper energy evaluation function
        grad[i] = 0.0  # Placeholder
    return grad