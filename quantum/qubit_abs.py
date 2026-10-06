"""A small, readable state-vector quantum simulator.

Design notes
------------
* Every gate is applied by building its full matrix and multiplying the state
  vector: `state = full_matrix @ state`. Nothing is mutated in place.
* `Qubit` objects are *handles*, not containers. A qubit owns no amplitudes, so
  passing one around (or storing it in a list) never copies quantum data.
* `op_hist` records every gate that is applied, which is what both `uncompute`
  and `to_openqasm3` read from.
"""

from contextlib import contextmanager
from typing import overload

import numpy as np
import numpy.random as rnd

from quantum.qasm import to_openqasm3


class Gate:
    """One recorded gate application: a matrix plus the qubits it touches.

    `inverted` marks a gate that was applied backwards (see `QuantumComputer.
    uncompute`), so the OpenQASM 3 export can write it as `inv @ ...`.
    """

    def __init__(self, matrix, controls: list[int], targets: list[int],
                 name: str | None = None, params: list[float] | None = None,
                 inverted: bool = False):
        self.matrix = np.asarray(matrix, dtype=complex)
        self.controls = list(controls)
        self.targets = list(targets)
        self.name = name
        self.params = list(params) if params else []
        self.inverted = inverted

    def __repr__(self):
        label = self.name if self.name else f"{self.matrix.shape[0]}x{self.matrix.shape[1]} matrix"
        direction = "inv @ " if self.inverted else ""
        qubits = ", ".join([str(i) for i in self.controls + self.targets])
        return f"<{direction}{label} on [{qubits}]>"

    def inverse(self) -> "Gate":
        """The same operation run backwards (the adjoint of a unitary)."""
        return Gate(
            self.matrix.conj().T,
            self.controls,
            self.targets,
            self.name,
            self.params,
            not self.inverted,
        )


def embedded_gate(gate_matrix: np.ndarray, num_qubits: int,
                  target_bits: list[int], control_bits: list[int]) -> np.ndarray:
    """Expand a small gate into the full 2**num_qubits operator it acts as.

    `target_bits` and `control_bits` are bit positions counted from the least
    significant bit. The gate only touches basis states where every control is
    1; every other basis state is left alone.
    """
    dimension = 1 << num_qubits
    basis = np.arange(dimension, dtype=np.int64)
    num_targets = len(target_bits)

    # Bit position of each target inside the gate's own little matrix.
    offsets = np.zeros(1 << num_targets, dtype=np.int64)
    for gate_bit, position in enumerate(target_bits):
        gate_value = (np.arange(1 << num_targets, dtype=np.int64) >> (num_targets - 1 - gate_bit)) & 1
        offsets |= gate_value << position

    # Basis states where all controls are 1 and all targets are 0.
    usable = np.ones(dimension, dtype=bool)
    for position in control_bits:
        usable &= ((basis >> position) & 1) == 1

    target_mask = 0
    for position in target_bits:
        target_mask |= 1 << position
    usable &= (basis & target_mask) == 0
    anchors = basis[usable]

    full_matrix = np.eye(dimension, dtype=complex)
    for row in range(1 << num_targets):
        for column in range(1 << num_targets):
            full_matrix[anchors + offsets[row], anchors + offsets[column]] = gate_matrix[row, column]

    return full_matrix


class CompositeGate:
    """A gate built from other operations rather than from one matrix.

    `body(qc, targets, controls)` applies whatever gates it needs; `name` and
    `params` describe it for OpenQASM 3 (its inner `apply_gate` calls are what
    actually end up in `op_hist`).
    """

    def __init__(self, name: str, body, params: list[float] | None = None):
        self.name = name
        self.body = body
        self.params = list(params) if params else []

    def __repr__(self):
        return f"<composite {self.name}>"

    def __call__(self, qc: "QuantumComputer", targets: list[int], controls: list[int]):
        self.body(qc, targets, controls)


def apply_local_gate(state: np.ndarray, gate_matrix: np.ndarray, num_qubits: int,
                     target_bits: list[int], control_bits: list[int]) -> np.ndarray:
    """Apply a gate to one or two targets without building a 2**n matrix.

    As everywhere else in the simulator, `gate_matrix` acts on the targets
    only and the controls say where that action is conditional.

    The state is viewed as one axis per qubit. Control axes are moved just
    before the target axes, so the flattened pair of axes reads as
    (controls, targets) with the first qubit of each group as the most
    significant bit. Only the block where every control reads 1 is multiplied
    by the gate. Cost is O(2**num_qubits) instead of O(4**num_qubits), which is
    what makes wide registers like Shor's practical.
    """
    num_targets, num_controls = len(target_bits), len(control_bits)
    tensor = state.reshape([2] * num_qubits)

    # Axis 0 of the tensor is the most significant bit, so flip the bit positions.
    target_axes = [num_qubits - 1 - bit for bit in target_bits]
    control_axes = [num_qubits - 1 - bit for bit in control_bits]

    # Controls occupy the axes before the targets; targets go last.
    first = num_qubits - num_controls - num_targets
    control_slots = list(range(first, first + num_controls))
    target_slots = list(range(first + num_controls, num_qubits))
    moved_axes = control_axes + target_axes

    tensor = np.moveaxis(tensor, moved_axes, control_slots + target_slots)
    blocks = tensor.reshape(-1, 1 << num_controls, 1 << num_targets).copy()

    # The last control block is the one with every control qubit reading 1.
    blocks[:, -1, :] = blocks[:, -1, :] @ gate_matrix.T

    tensor = blocks.reshape([2] * num_qubits)
    tensor = np.moveaxis(tensor, control_slots + target_slots, moved_axes)
    return tensor.reshape(-1, 1)


class Qubit:
    """A handle to one physical qubit. Holds an index, never any amplitudes."""

    def __init__(self, computer: "QuantumComputer", index: int, name: str = "qubit"):
        self.computer = computer
        self.index = index
        self.name = name
        self.is_measured = False
        self._true_val = None

    def __repr__(self):
        return f"{self.name}({self.index})"

    @classmethod
    def from_list_str(cls, computer: "QuantumComputer", name_list: list[str]) -> list["Qubit"]:
        return [cls(computer, index, name) for index, name in enumerate(name_list)]

    def is_alive(self) -> bool:
        return not self.is_measured

    def _assert_alive(self):
        if not self.is_alive():
            raise ValueError(f"Qubit {self.name} is locked after measurement (no gates can be applied).")

    def apply_gate(self, gate_matrix: np.ndarray):
        self._assert_alive()
        self.computer.apply_gate(gate_matrix, target_qubits=[self])

    def probability_of_one(self) -> float:
        """Chance this qubit reads 1 if we measured the whole computer now."""
        amplitudes = self.computer.state_vector[:, 0]
        bit = self.computer.bit_position(self.index)
        return float(np.sum(np.abs(amplitudes[(np.arange(len(amplitudes)) >> bit) & 1 == 1]) ** 2))

    def measure(self) -> int:
        """Measure this qubit, collapsing the state onto the outcome."""
        self._assert_alive()
        probability = np.clip(self.probability_of_one(), 0.0, 1.0)

        self.is_measured = True
        self._true_val = int(rnd.choice([0, 1], p=[1 - probability, probability]))

        bit = self.computer.bit_position(self.index)
        basis = np.arange(self.computer.dimension)
        collapse_to = ((basis >> bit) & 1) == self._true_val

        collapsed = np.zeros_like(self.computer.state_vector)
        collapsed[collapse_to, 0] = self.computer.state_vector[collapse_to, 0]

        norm = np.linalg.norm(collapsed)
        self.computer.state_vector = collapsed / norm if norm > 0 else collapsed

        return self._true_val


class QuantumComputer:
    def __init__(self, num_qubits: int, name: str = "QC1", qubit_names: list[str] | None = None):
        self.num_qubits = num_qubits
        self.name = name
        self.state_vector = np.zeros((self.dimension, 1), dtype=complex)
        self.state_vector[0, 0] = 1.0
        self.op_hist: list[Gate] = []
        self._recording = True

        if qubit_names is None:
            self.qubits: list[Qubit] = [Qubit(self, i, f"q{i}") for i in range(num_qubits)]
        else:
            if len(qubit_names) != num_qubits:
                raise ValueError(
                    f"The number of qubit names ({len(qubit_names)}) must match "
                    f"the number of qubits ({self.num_qubits})."
                )
            self.qubits = Qubit.from_list_str(self, qubit_names)

    @property
    def dimension(self) -> int:
        return 1 << self.num_qubits

    def bit_position(self, qubit_index: int) -> int:
        """Turn a qubit index into a bit position (q0 is the most significant bit)."""
        return self.num_qubits - 1 - qubit_index

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
        if isinstance(key, int):
            index = key + self.num_qubits if key < 0 else key
            if not 0 <= index < self.num_qubits:
                raise IndexError(f"Qubit index {key} is out of range for {self.num_qubits} qubits.")
            return self.qubits[index]
        if isinstance(key, str):
            for qubit in self.qubits:
                if qubit.name == key:
                    return qubit
            raise KeyError(f"No qubit found with name '{key}'.")
        if isinstance(key, slice):
            start, stop, step = key.indices(self.num_qubits)
            return [self.qubits[i] for i in range(start, stop, step)]
        raise TypeError("Key must be a Qubit, integer index, string name, or slice.")

    def __repr__(self):
        listed = "\n  └─ ".join(str(qubit) for qubit in self.qubits)
        return f"System: {self.name} ({self.num_qubits} Qubits)\n  └─ {listed}"

    def index_of(self, qubit: Qubit | int | str) -> int:
        """Accept a handle, an index, or a name wherever a qubit is expected."""
        if isinstance(qubit, Qubit):
            return qubit.index
        resolved = self[qubit]
        if isinstance(resolved, Qubit):
            return resolved.index
        raise TypeError("Expected a single qubit, not a list of qubits.")

    def apply_gate(
        self,
        gate_matrix: np.ndarray,
        target_qubits: list[Qubit | int | str],
        control_qubits: list[Qubit | int | str] | None = None,
        qasm_name: str | None = None,
        qasm_params: list[float] | None = None,
        log_history: bool = True,
    ):
        """Apply a gate, optionally controlled, to the given qubits."""
        gate = Gate(
            gate_matrix,
            controls=[self.index_of(qubit) for qubit in (control_qubits or [])],
            targets=[self.index_of(qubit) for qubit in target_qubits],
            name=qasm_name,
            params=qasm_params,
        )

        self._run(gate)
        if log_history and self._recording:
            self.op_hist.append(gate)

    def apply(
        self,
        operation: np.ndarray | CompositeGate,
        target_qubits: list[Qubit | int | str],
        control_qubits: list[Qubit | int | str] | None = None,
    ):
        """Apply a matrix gate, or a composite operation built from other gates.

        Both take the same operands, so `qc.apply(gm.h_MAT, [0])` and
        `qc.apply(crk(3), [0], control_qubits=[1])` read the same way.
        """
        if callable(operation):
            operation(
                self,
                [self.index_of(qubit) for qubit in target_qubits],
                [self.index_of(qubit) for qubit in (control_qubits or [])],
            )
        else:
            self.apply_gate(operation, target_qubits, control_qubits)

    def _run(self, gate: Gate):
        """Advance the state by one gate, without touching the history."""
        for index in gate.controls + gate.targets:
            self.qubits[index]._assert_alive()

        control_bits = [self.bit_position(index) for index in gate.controls]
        target_bits = [self.bit_position(index) for index in gate.targets]

        if len(target_bits) <= 2:
            self.state_vector = apply_local_gate(
                self.state_vector, gate.matrix, self.num_qubits, target_bits, control_bits
            )
        else:
            full_matrix = embedded_gate(gate.matrix, self.num_qubits, target_bits, control_bits)
            self.state_vector = full_matrix @ self.state_vector

    @contextmanager
    def suppress_history(self):
        """Run a block of gates without recording them (e.g. oracle internals)."""
        was_recording = self._recording
        self._recording = False
        try:
            yield self
        finally:
            self._recording = was_recording

    def checkpoint(self) -> int:
        """Mark the current point in the history; pass it back to `uncompute`."""
        return len(self.op_hist)

    def uncompute(self, target_qubits: list[Qubit | int | str], checkpoint: int = 0):
        """Reverse every recorded gate since `checkpoint` that touches these qubits.

        Each reversal is *appended* to the history as the inverse gate rather
        than erasing the original, so `op_hist` stays a truthful record of
        everything that ran and `to_openqasm3` exports exactly that.
        """
        if not self.op_hist:
            raise ValueError("No operations to uncompute.")

        wanted = {self.index_of(qubit) for qubit in target_qubits}
        was_recording, self._recording = self._recording, True

        try:
            for gate in reversed(self.op_hist[checkpoint:]):
                touched = set(gate.controls + gate.targets)
                if not touched & wanted:
                    continue
                if any(self.qubits[index].is_measured for index in touched):
                    raise ValueError("Cannot uncompute after measurement of involved qubits.")

                inverse = gate.inverse()
                self._run(inverse)
                self.op_hist.append(inverse)
        finally:
            self._recording = was_recording

    def sample(self, circuit_func, num_samples: int = 1000) -> dict[str, int]:
        """Reset, run the circuit, and return bitstring -> count."""
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")

        self.state_vector = np.zeros((self.dimension, 1), dtype=complex)
        self.state_vector[0, 0] = 1.0
        self.op_hist = []
        for qubit in self.qubits:
            qubit.is_measured = False
            qubit._true_val = None

        circuit_func(self)

        probabilities = np.abs(self.state_vector.flatten()) ** 2
        total = probabilities.sum()
        if total > 0:
            probabilities /= total

        draws = rnd.choice(self.dimension, size=num_samples, p=probabilities)
        counts = np.bincount(draws, minlength=self.dimension)

        return {f"{i:0{self.num_qubits}b}": int(count) for i, count in enumerate(counts) if count}

    def to_openqasm3(self) -> str:
        """Render the recorded history as an OpenQASM 3 program."""
        return to_openqasm3(self.num_qubits, self.name, self.op_hist)
