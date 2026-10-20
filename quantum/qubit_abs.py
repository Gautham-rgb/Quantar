"""A small, readable state-vector quantum simulator.

Design notes
------------
* One or two qubit gates go through a numba kernel that reads each amplitude
  once and writes it once; bigger gates fall back to `embedded_gate`, which
  builds the full 2**n operator. Nothing is ever mutated in place.
* A run of diagonal gates (the controlled phases of a QFT) is applied in one
  fused pass over the amplitudes, and a run of SWAPs (a bit-reversal layer)
  in a single permutation pass, instead of one pass per gate.
* A whole circuit can be used as a single gate: see `CircuitGate`.
* `Qubit` objects are *handles*, not containers. A qubit owns no amplitudes, so
  passing one around (or storing it in a list) never copies quantum data.
* `op_hist` records every gate that is applied, which is what both `uncompute`
  and `to_openqasm3` read from.
"""

from contextlib import contextmanager
from typing import overload

import numpy as np
import numpy.random as rnd
from typing import Callable

from quantum.qasm import to_openqasm3
import quantum.gate_matrix as gm
from quantar.operators import get_global_operator, normalize_kind

from quantum.gate_abs import Gate, CompositeGate, CircuitGate

from quantum.basic_funcs import (
    bit_mask, embedded_gate, apply_local_gate, _is_diagonal_local, _is_swap_local,
)
from quantum.kernels import (
    HAVE_NUMBA, fused_diagonal_kernel, fused_diagonal_numpy,
    swap_run_kernel, swap_run_numpy, sample_counts,
)

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
        self._scratch: np.ndarray | None = None
        self._custom_operators: dict[str, np.ndarray | Callable] = {}

        if qubit_names is None:
            self.qubits: list[Qubit] = [Qubit(self, i, f"q{i}") for i in range(num_qubits)]
        else:
            if len(qubit_names) != num_qubits:
                raise ValueError(
                    f"The number of qubit names ({len(qubit_names)}) must match "
                    f"the number of qubits ({self.num_qubits})."
                )
            self.qubits = Qubit.from_list_str(self, qubit_names)

    def register_operator(self, name: str, matrix: np.ndarray | Callable) -> None:
        """Register a custom operator by name.

        Args:
            name: Name to reference the operator (e.g., "my_gate")
            matrix: Unitary matrix (numpy array) or callable that returns a matrix
                   Callable signature: callable(*params) -> np.ndarray
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Operator name must be a non-empty string")
        if name in self._custom_operators:
            raise ValueError(f"Operator '{name}' already registered")
        self._custom_operators[name] = matrix

    def get_operator(self, name: str) -> np.ndarray | Callable | None:
        """Retrieve a custom operator by name."""
        return self._custom_operators.get(name)

    def list_operators(self) -> list[str]:
        """List all registered custom operator names."""
        return list(self._custom_operators.keys())

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
        listed = "\n  +- ".join(str(qubit) for qubit in self.qubits)
        return f"System: {self.name} ({self.num_qubits} Qubits)\n  +- {listed}"

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
        log_history: bool | None = None,
    ):
        """Apply a gate, optionally controlled, to the given qubits."""
        gate = self.gate(
            gate_matrix, target_qubits, control_qubits, qasm_name, qasm_params
        )

        record = self._recording
        if log_history is not None:
            record = bool(log_history)

        self._run(gate)
        if record and self._recording:
            self.op_hist.append(gate)

    def gate(
        self,
        gate_matrix: np.ndarray,
        target_qubits: list[Qubit | int | str],
        control_qubits: list[Qubit | int | str] | None = None,
        qasm_name: str | None = None,
        qasm_params: list[float] | None = None,
    ) -> Gate:
        """Build a gate record without applying it.

        This is how programs are written: gather records, then hand the list to
        `qc.apply([...])`, which runs them with diagonal runs fused into a
        single pass over the state.
        """
        return Gate(
            gate_matrix,
            controls=[self.index_of(qubit) for qubit in (control_qubits or [])],
            targets=[self.index_of(qubit) for qubit in target_qubits],
            name=qasm_name,
            params=qasm_params,
        )

    def apply(
        self,
        operation: np.ndarray | CompositeGate | CircuitGate | Gate | list[Gate] | str,
        target_qubits: list[Qubit | int | str] | None = None,
        control_qubits: list[Qubit | int | str] | None = None,
    ):
        """Apply a matrix, a gate record, a program, a custom operator name, or a composite operation.

        All forms read the same way, so `qc.apply(gm.h_MAT, [0])`,
        `qc.apply(crk(3), [0], control_qubits=[1])` and
        `qc.apply(qft_gate(3), qc[0:3])` sit side by side. A program
        (`qc.apply([gate1, gate2, ...])`) picks its own qubits, so it takes no
        target operand; extra `control_qubits` condition every gate in it.

        If `operation` is a string, it's looked up in the custom operator registry
        (both instance and global registries).
        """
        if isinstance(operation, str):
            # Look up custom operator in the global registry, then the instance one.
            op, kind = get_global_operator(operation)
            if op is None:
                op = self.get_operator(operation)
                if op is None:
                    raise ValueError(f"Unknown operator: '{operation}'")
                kind = getattr(op, "_kind", "matrix")
            kind = normalize_kind(kind or "matrix")
            requires_instance = getattr(op, "_requires_instance", False)

            if kind == "nullary":
                if not callable(op):
                    return op
                return op(self) if requires_instance else op()

            if kind == "matrix":
                if callable(op):
                    matrix = op(self) if requires_instance else op()
                else:
                    matrix = op
                if not isinstance(matrix, np.ndarray):
                    raise ValueError(f"Operator '{operation}' must return a numpy array")
                if target_qubits is None:
                    raise ValueError("Custom operator requires target qubits")
                self.apply_gate(matrix, target_qubits, control_qubits, qasm_name=operation)
                return

            if kind in ("unary_pre", "unary_post"):
                if target_qubits is None or len(target_qubits) != 1:
                    raise ValueError(
                        f"Unary operator '{operation}' requires exactly one target qubit"
                    )
                if not callable(op):
                    raise ValueError(f"Unary operator '{operation}' is not callable")
                target = self.index_of(target_qubits[0])
                return op(self, target) if requires_instance else op(target)

            if kind == "binary":
                if target_qubits is None or len(target_qubits) != 2:
                    raise ValueError(
                        f"Binary operator '{operation}' requires exactly two target qubits (control, target)"
                    )
                if not callable(op):
                    raise ValueError(f"Binary operator '{operation}' is not callable")
                left = self.index_of(target_qubits[0])
                right = self.index_of(target_qubits[1])
                return op(self, left, right) if requires_instance else op(left, right)

            if kind == "ternary":
                if target_qubits is None or len(target_qubits) != 3:
                    raise ValueError(
                        f"Ternary operator '{operation}' requires exactly three target qubits"
                    )
                if not callable(op):
                    raise ValueError(f"Ternary operator '{operation}' is not callable")
                operands = [self.index_of(qubit) for qubit in target_qubits]
                return op(self, *operands) if requires_instance else op(*operands)

            # kind == "callable": free-form, gets the target/control index lists.
            if target_qubits is None:
                raise ValueError("Callable operator requires target qubits")
            if not callable(op):
                raise ValueError(f"Callable operator '{operation}' is not callable")
            targets = [self.index_of(qubit) for qubit in target_qubits]
            controls = [self.index_of(qubit) for qubit in (control_qubits or [])]
            return op(self, targets, controls) if requires_instance else op(targets, controls)

        if isinstance(operation, (list, tuple)):
            gates = list(operation)
            if target_qubits:
                raise ValueError("A program names its own qubits; drop the target operand.")
            if control_qubits:
                extra = [self.index_of(qubit) for qubit in control_qubits]
                identity = list(range(self.num_qubits))
                gates = [gate.remap(identity, extra) for gate in gates]
            self._apply_program(gates)
            return

        if isinstance(operation, Gate):
            self._apply_program([operation])
            return

        if target_qubits is None:
            raise ValueError("This operation needs target qubits.")

        if callable(operation):
            operation(
                self,
                [self.index_of(qubit) for qubit in target_qubits],
                [self.index_of(qubit) for qubit in (control_qubits or [])],
            )
        else:
            self.apply_gate(operation, target_qubits, control_qubits)

    def _validate_gate(self, gate: Gate):
        """Check a gate against the register before any amplitude is touched."""
        if not isinstance(gate, Gate):
            raise TypeError(f"Expected a Gate, got {type(gate).__name__}.")

        for index in gate.controls + gate.targets:
            self.qubits[index]._assert_alive()

        if set(gate.controls) & set(gate.targets):
            raise ValueError("Control and target qubits must be disjoint.")

        side = 1 << len(gate.targets)
        if gate.matrix.shape != (side, side):
            raise ValueError(
                f"A gate on {len(gate.targets)} target(s) needs a {side}x{side} matrix, "
                f"got {gate.matrix.shape}."
            )

    def _apply_program(self, program: list[Gate]):
        """Run a list of gates, batching the runs that can share a pass.

        Consecutive diagonal gates become one phase pass and consecutive
        disjoint SWAPs one permutation pass; everything else goes through
        `_run`. Only the execution is batched: `op_hist` receives the
        individual gates, so `uncompute` and `to_openqasm3` still see `cp`
        after `cp`.
        """
        gates = list(program)
        for gate in gates:
            self._validate_gate(gate)

        record = self._recording
        index = 0
        while index < len(gates):
            gate = gates[index]

            if _is_diagonal_local(gate):
                run_end = index + 1
                while run_end < len(gates) and _is_diagonal_local(gates[run_end]):
                    run_end += 1
                batch = gates[index:run_end]
                if len(batch) == 1:
                    self._run(batch[0])
                else:
                    self._apply_diagonal_run(batch)

            elif _is_swap_local(gate):
                # Overlapping swaps do not commute, so the run stops at the
                # first shared qubit.
                touched = set(gate.targets)
                run_end = index + 1
                while (
                    run_end < len(gates)
                    and _is_swap_local(gates[run_end])
                    and not touched & set(gates[run_end].targets)
                ):
                    touched.update(gates[run_end].targets)
                    run_end += 1
                batch = gates[index:run_end]
                if len(batch) == 1:
                    self._run(batch[0])
                else:
                    self._apply_swap_run(batch)

            else:
                self._run(gate)
                run_end = index + 1
                batch = [gate]

            if record:
                self.op_hist.extend(batch)
            index = run_end

    def _apply_swap_run(self, gates: list[Gate]):
        """Apply several disjoint SWAPs in one permutation pass."""
        pairs = np.array(
            [
                [self.bit_position(gate.targets[0]), self.bit_position(gate.targets[1])]
                for gate in gates
            ],
            dtype=np.int64,
        )

        flat = self.state_vector.reshape(-1)
        if self._scratch is None or self._scratch.shape != self.state_vector.shape:
            self._scratch = np.empty_like(self.state_vector)
        out = self._scratch.reshape(-1) #type: ignore

        if HAVE_NUMBA:
            swap_run_kernel(out, flat, pairs)
        else:
            swap_run_numpy(out, flat, pairs)

        previous_state = self.state_vector
        self.state_vector = out.reshape(-1, 1)
        self._scratch = previous_state

    def _apply_diagonal_run(self, gates: list[Gate]):
        """Apply several diagonal single-target gates in one pass over the state."""
        control_masks = np.array(
            [bit_mask(self.bit_position(index) for index in gate.controls) for gate in gates],
            dtype=np.int64,
        )
        target_bits = np.array(
            [self.bit_position(gate.targets[0]) for gate in gates], dtype=np.int64
        )
        diag0 = np.array([gate.matrix[0, 0] for gate in gates], dtype=complex)
        diag1 = np.array([gate.matrix[1, 1] for gate in gates], dtype=complex)

        flat = self.state_vector.reshape(-1)
        if self._scratch is None or self._scratch.shape != self.state_vector.shape:
            self._scratch = np.empty_like(self.state_vector)
        out = self._scratch.reshape(-1) #type: ignore

        if HAVE_NUMBA:
            fused_diagonal_kernel(out, flat, control_masks, target_bits, diag0, diag1)
        else:
            fused_diagonal_numpy(out, flat, control_masks, target_bits, diag0, diag1)

        previous_state = self.state_vector
        self.state_vector = out.reshape(-1, 1)
        self._scratch = previous_state

    def _run(self, gate: Gate):
        """Advance the state by one gate, without touching the history."""
        self._validate_gate(gate)

        control_bits = [self.bit_position(index) for index in gate.controls]
        target_bits = [self.bit_position(index) for index in gate.targets]

        if self._scratch is None or self._scratch.shape != self.state_vector.shape:
            self._scratch = np.empty_like(self.state_vector)

        # The result lands in the scratch buffer and the two swap, so the
        # amplitudes are only ever written into a buffer that is already warm.
        previous_state = self.state_vector
        if len(target_bits) <= 2:
            # One fused kernel pass over the amplitudes; the kernel is correct
            # for every register size and beats the einsum path in practice.
            self.state_vector = apply_local_gate(
                self.state_vector, gate.matrix, self.num_qubits, target_bits,
                control_bits, out=self._scratch,
            )
        else:
            full_matrix = embedded_gate(gate.matrix, self.num_qubits, target_bits, control_bits)
            self.state_vector = np.matmul(full_matrix, self.state_vector, out=self._scratch)
        self._scratch = previous_state

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

    def qft(self, qubits: list[Qubit | int | str], *, inverse: bool = False,
            record: bool | None = None) -> "QuantumComputer":
        """Apply a Quantum Fourier Transform to the given qubits.

        The transform is built from Hadamards, controlled phase shifts and a
        bit-reversing swap layer (see `stl.quantum_algs.shors_alg`); runs of
        diagonal phases are fused into one pass over the state. `qubits[0]` is
        treated as the most significant bit.

        Args:
            qubits: The qubits to transform (MSB first).
            inverse: If True, apply the inverse QFT (IQFT).
            record: Whether to record in history (default: follows _recording).

        Returns:
            Self, for chaining.
        """
        targets = [self.index_of(q) for q in qubits]
        if not targets:
            return self

        from stl.quantum_algs.shors_alg import qft_program, iqft_program

        program = (iqft_program if inverse else qft_program)(self, targets)

        if record is None:
            self.apply(program)
        elif record:
            was_recording = self._recording
            self._recording = True
            try:
                self.apply(program)
            finally:
                self._recording = was_recording
        else:
            with self.suppress_history():
                self.apply(program)

        return self

    def iqft(self, qubits: list[Qubit | int | str], **kwargs) -> "QuantumComputer":
        """Apply a fast Inverse Quantum Fourier Transform.

        Equivalent to `qft(qubits, inverse=True)`.
        """
        return self.qft(qubits, inverse=True, **kwargs)

    def measure_pauli(self, pauli: str, qubits: list[Qubit | int | str],
                      shots: int = 1000) -> float:
        """Estimate the expectation value of a Pauli string.

        Each qubit is rotated into the Z basis, sampled, then rotated back, so
        the circuit is left unchanged. The rotations are not recorded in
        `op_hist` -- they are part of the measurement, not the program.

        Args:
            pauli: Pauli string like 'X', 'Y', 'Z', 'XX', 'ZZ', 'XY', 'XYZ', etc.
            qubits: List of qubits to measure (must match length of pauli string)
            shots: Number of samples used for the estimate.

        Returns:
            Expectation value <psi|pauli|psi> as a float.
        """
        targets = [self.index_of(q) for q in qubits]
        if len(targets) != len(pauli):
            raise ValueError(f"Pauli string length ({len(pauli)}) must match number of qubits ({len(targets)})")
        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        with self.suppress_history():
            # Rotate each qubit into the Z basis.
            for idx, p in zip(targets, pauli):
                if p == 'X':
                    self.apply_gate(gm.h_MAT(), [self.qubits[idx]])
                elif p == 'Y':
                    # Y basis: S† then H maps Y -> Z.
                    self.apply_gate(gm.sdg_MAT(), [self.qubits[idx]])
                    self.apply_gate(gm.h_MAT(), [self.qubits[idx]])
                elif p in ('Z', 'I'):
                    pass  # already in the computational basis
                else:
                    raise ValueError(f"Unknown Pauli operator: {p}")

            probs = np.abs(self.state_vector.flatten()) ** 2
            total = probs.sum()
            if total > 0:
                probs /= total

            parity_values = []
            for _ in range(shots):
                outcome = int(rnd.choice(self.dimension, p=probs))
                parity = 1.0
                for idx in targets:
                    if (outcome >> self.bit_position(idx)) & 1:
                        parity = -parity
                parity_values.append(parity)

            # Undo the basis change (the inverse of H·S† is S·H: H first, then S).
            for idx, p in zip(targets, pauli):
                if p == 'X':
                    self.apply_gate(gm.h_MAT(), [self.qubits[idx]])
                elif p == 'Y':
                    self.apply_gate(gm.h_MAT(), [self.qubits[idx]])
                    self.apply_gate(gm.s_MAT(), [self.qubits[idx]])
                elif p in ('Z', 'I'):
                    pass

        return float(np.mean(parity_values))

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

        probabilities = np.abs(self.state_vector.ravel()) ** 2
        total = probabilities.sum()
        if total > 0:
            probabilities /= total

        counts = sample_counts(probabilities, num_samples, self.dimension)

        return {f"{i:0{self.num_qubits}b}": int(count) for i, count in enumerate(counts) if count}

    def sample_parallel(self, circuit_func: Callable, num_samples: int = 1000, **circuit_func_kwargs):
        """Reset the computer and sample measurement outcomes in parallel."""
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")

        self.state_vector = np.zeros((self.dimension, 1), dtype=complex)
        self.state_vector[0, 0] = 1.0
        self.op_hist = []
        for qubit in self.qubits:
            qubit.is_measured = False
            qubit._true_val = None

        circuit_func(self, **circuit_func_kwargs)
        probabilities = np.abs(self.state_vector.ravel()) ** 2
        total = probabilities.sum()
        if total > 0:
            probabilities /= total

        counts = sample_counts(probabilities, num_samples, self.dimension)
        return {f"{i:0{self.num_qubits}b}": int(count) for i, count in enumerate(counts) if count}


    def to_openqasm3(self) -> str:
        """Render the recorded history as an OpenQASM 3 program."""
        return to_openqasm3(self.num_qubits, self.name, self.op_hist)
