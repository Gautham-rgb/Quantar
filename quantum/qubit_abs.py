import numpy.random as rnd
import numpy as np
from typing import overload

import quantum.gate_matrix as gm


def _embedded_gate(gate_matrix: np.ndarray, num_qubits: int,
                   target_positions: list[int], control_positions: list[int]) -> np.ndarray:
    dim = 1 << num_qubits
    indices = np.arange(dim, dtype=np.int64)

    num_targets = len(target_positions)

    target_mask = 0
    for pos in target_positions:
        target_mask |= 1 << pos

    offset = np.zeros(1 << num_targets, dtype=np.int64)
    local_indices = np.arange(1 << num_targets, dtype=np.int64)
    for bit, pos in enumerate(target_positions):
        bit_val = (local_indices >> (num_targets - 1 - bit)) & 1
        offset |= (bit_val << pos)

    controlled_mask = np.ones(dim, dtype=bool)
    for pos in control_positions:
        controlled_mask &= ((indices >> pos) & 1) == 1

    base_states = indices[controlled_mask & ((indices & target_mask) == 0)]

    full_matrix = np.eye(dim, dtype=complex)
    for row in range(1 << num_targets):
        for column in range(1 << num_targets):
            full_matrix[base_states + offset[row], base_states + offset[column]] = gate_matrix[row, column]

    return full_matrix


class Qubit:
    def __init__(self, computer: QuantumComputer, index: int, name: str = "qubit"):
        self.computer = computer
        self.index = index
        self.name = name
        self.is_measured = False
        self._true_val = None

    def __repr__(self):
        return f"{self.name}({self.index})"

    @classmethod
    def from_list_str(cls, computer: QuantumComputer, name_list: list[str]):
        qubits = []
        for i in range(len(name_list)):
            qubits.append(cls(computer, i, name_list[i]))
        return qubits

    def is_alive(self):
        return not self.is_measured

    def _assert_alive(self):
        if not self.is_alive():
            raise ValueError(f"Qubit {self.name} is locked after measurement (no gates can be applied).")

    def apply_gate(self, gate_matrix: np.ndarray):
        self._assert_alive()
        self.computer.apply_gate(gate_matrix, target_qubits=[self])

    def measure(self):
        self._assert_alive()
        if not self.is_alive():
            return self._true_val

        self.is_measured = True
        self._true_val = rnd.choice([0, 1], p=[1 - self.prob_1, self.prob_1])
        self.prob_1 = 1.0 if self._true_val == 1 else 0.0
        return self._true_val

class QuantumComputer:
    def __init__(self, num_qubits: int, name: str = "QC1", qubit_names: list[str] | None = None):
        self.num_qubits = num_qubits
        self.name = name
        self.state_vector = np.zeros((2 ** num_qubits, 1), dtype=complex)
        self.state_vector[0, 0] = 1.0

        if qubit_names is None:
            self.qubits: list[Qubit] = [Qubit(self, index=i, name=f"q{i}") for i in range(num_qubits)]
        else:
            if len(qubit_names) != num_qubits:
                raise ValueError(f"The number of qubit names ({len(qubit_names)}) must match the number of qubits ({self.num_qubits}).")
            self.qubits = Qubit.from_list_str(self, qubit_names)

    @overload
    def __getitem__(self, key: Qubit) -> Qubit: ...

    @overload
    def __getitem__(self, key: int) -> Qubit: ...

    @overload
    def __getitem__(self, key: str) -> Qubit: ...

    @overload
    def __getitem__(self, key: slice) -> list[Qubit]: ...

    def __getitem__(self, key):
        if isinstance(key, Qubit):
            return key
        elif isinstance(key, int):
            idx = key
            if idx < 0:
                idx += self.num_qubits
            if idx < 0 or idx >= self.num_qubits:
                raise IndexError(f"Qubit index {key} is out of range for a quantum computer with {self.num_qubits} qubits.")
            return self.qubits[idx]
        elif isinstance(key, str):
            for qubit in self.qubits:
                if qubit.name == key:
                    return qubit
            raise KeyError(f"No qubit found with name '{key}'.")
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.num_qubits)
            return [self.qubits[i] for i in range(start, stop, step)]
        else:
            raise TypeError("Key must be a Qubit, integer index, string name, or slice.")

    def __repr__(self):
        qubit_lines = "\n  └─ ".join([str(qubit) for qubit in self.qubits])
        return f"System: {self.name} ({self.num_qubits} Qubits)\n  └─ {qubit_lines}"

    def apply_gate(
        self,
        gate_matrix: np.ndarray,
        target_qubits: list[Qubit | int | str],
        control_qubits: list[Qubit | int | str] | None = None,
    ):
        def get_index(qubit: Qubit | int | str) -> int:
            if isinstance(qubit, Qubit):
                return qubit.index
            result = self[qubit]
            if isinstance(result, Qubit):
                return result.index
            raise TypeError("Target/control must resolve to a single qubit.")

        crtl_list = [get_index(q) for q in (control_qubits or [])]
        targ_list = [get_index(q) for q in target_qubits]

        crtl_pos = [self.num_qubits - 1 - idx for idx in crtl_list]
        targ_pos = [self.num_qubits - 1 - idx for idx in targ_list]

        matrix = np.asarray(gate_matrix, dtype=complex)
        full_matrix = _embedded_gate(matrix, self.num_qubits, targ_pos, crtl_pos)
        self.state_vector = full_matrix @ self.state_vector

    def sample(self, circuit_func, num_samples: int = 1000) -> dict[str, int]:
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")

        self.state_vector = np.zeros((1 << self.num_qubits, 1), dtype=complex)
        self.state_vector[0, 0] = 1.0
        for qubit in self.qubits:
            qubit.is_measured = False
            qubit._true_val = None

        circuit_func(self)

        probabilities = np.abs(self.state_vector.flatten()) ** 2
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities /= prob_sum

        samples = rnd.choice(1 << self.num_qubits, size=num_samples, p=probabilities)
        counts = np.bincount(samples, minlength=1 << self.num_qubits)

        non_zero = np.nonzero(counts)[0]
        return {
            f"{i:0{self.num_qubits}b}": int(counts[i])
            for i in non_zero
        }
