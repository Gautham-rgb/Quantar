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
import re
import tokenize
import io
from typing import Any
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco
    prange = range

from quantum.qasm import to_openqasm3
import quantum.gate_matrix as gm
# ─── Local operator registry (for @operator decorator) ───
_global_operators: dict[str, Any] = {}
_operator_metadata: dict[str, dict[str, Any]] = {}

from quantar.operators import (
    get_global_operator, get_global_operator_matrix,
    get_operator_kind, get_operator_requires_instance,
    list_operators, apply_operator,
)


def operator(name: str, kind: str = "matrix"):
    """Decorator to register a custom operator with an arbitrary name.

    The name can be any string: capital letters, symbols, unicode, etc.
    Works as a function decorator, method decorator, class decorator,
    or with any Python/Quantar class.

    Args:
        name: Operator name (any string: "H", "X", "CNOT", "<->", "⟂", etc.)
        kind: Operator kind - "matrix" (default, returns numpy array), 
              "unary" (callable taking target), "binary" (callable taking control, target),
              "callable" (generic callable, inspected at runtime)

    Usage:
        @operator("H")
        def hadamard():
            return gm.h_MAT

        @operator("CNOT", kind="binary")
        def cnot(self, control, target):
            self.qc.apply_cnot(control, target)

        @operator("H", kind="unary")
        def hadamard(self, target):
            self.qc.apply_h(target)

        @operator("PRINT", kind="callable")
        def print_state(qc):
            print(qc.state)
    """
    def decorator(func: Callable):
        if not isinstance(name, str) or not name:
            raise ValueError("Operator name must be a non-empty string")
        
        if kind not in ("matrix", "unary", "binary", "callable"):
            raise ValueError(f"Invalid kind: {kind}. Must be 'matrix', 'unary', 'binary', or 'callable'")
        
        # Handle different function types
        if isinstance(func, (staticmethod, classmethod)):
            func = func.__func__
        
        # For matrix kind, try to call to get matrix immediately
        if kind == "matrix":
            try:
                matrix = func()
            except TypeError as e:
                if "missing 1 required positional argument: 'self'" in str(e):
                    def wrapper(instance):
                        return func(instance)
                    wrapper._quantar_operator_name = name
                    wrapper._quantar_is_operator = True
                    wrapper._quantar_requires_instance = True
                    wrapper._quantar_kind = "matrix"
                    _global_operators[name] = wrapper
                    func._quantar_operator_name = name
                    func._quantar_is_operator = True
                    func._quantar_requires_instance = True
                    func._quantar_kind = "matrix"
                    return func
                raise
            
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"Operator function '{name}' must return a numpy array")
            
            _global_operators[name] = matrix
            func._quantar_operator_name = name
            func._quantar_is_operator = True
            func._quantar_requires_instance = False
            func._quantar_kind = "matrix"
            return func
        
        # For unary/binary/callable kinds, store the callable directly
        # These will be called at execution time with appropriate arguments
        func._quantar_operator_name = name
        func._quantar_is_operator = True
        func._quantar_requires_instance = "self" in func.__code__.co_varnames
        func._quantar_kind = kind
        
        # For callable kind, we don't try to call it now
        _global_operators[name] = func
        
        return func
    
    return decorator


def get_global_operator(name: str):
    """Retrieve a globally registered operator by name.
    
    Returns a tuple of (operator, kind) where kind is one of:
    'matrix', 'unary', 'binary', 'callable', or None if not found.
    """
    op = _global_operators.get(name)
    if op is None:
        return None, None
    kind = getattr(op, '_quantar_kind', 'matrix')
    return op, kind


def get_global_operator_matrix(name: str):
    """Retrieve a globally registered matrix operator by name."""
    op, kind = get_global_operator(name)
    if kind == 'matrix':
        return op
    return None


def list_global_operators() -> list[str]:
    """List all globally registered operator names."""
    return list(_global_operators.keys())


def preprocess_quantar_source(source: str) -> str:
    """Pre-process Quantar source to handle custom operators.
    
    Replaces registered custom operators with space-padded versions
    so Python's tokenizer treats them as single OP tokens.
    """
    operators = sorted(list_global_operators(), key=len, reverse=True)
    if not operators:
        return source
    
    # Escape special regex chars and build pattern (longest first)
    escaped = [re.escape(op) for op in operators]
    pattern = '|'.join(escaped)
    
    def replace_op(match):
        op = match.group(0)
        # Replace with space-padded version so tokenizer sees it as single OP
        return f' {op} '
    
    return re.sub(pattern, replace_op, source)


def make_quantar_tokenizer(source: str):
    """Create a tokenizer for Quantar source with custom operator support."""
    processed = preprocess_quantar_source(source)
    return tokenize.generate_tokens(io.StringIO(processed).readline)


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

    def remap(self, mapping: list[int], extra_controls: list[int] = []) -> "Gate":
        """The same gate with its qubits rewritten through `mapping`.

        This is how a program built on one register is replayed onto another
        (see `CircuitGate`); `extra_controls` are appended, which is how a
        control on the outer gate conditions every gate inside it.
        """
        return Gate(
            self.matrix,
            [mapping[index] for index in self.controls] + list(extra_controls),
            [mapping[index] for index in self.targets],
            self.name,
            self.params,
            self.inverted,
        )


def bit_mask(positions) -> int:
    """One integer whose set bits are exactly `positions` (1 << position each)."""
    mask = 0
    for position in positions:
        mask |= 1 << position
    return mask


def gate_offsets(target_bits: list[int]) -> np.ndarray:
    """The physical bit each value of the gate's own little index sets.

    `target_bits[0]` is the gate matrix's most significant bit, so entry `v`
    of the result is the basis offset of gate column/row `v`.
    """
    num_targets = len(target_bits)
    offsets = np.zeros(1 << num_targets, dtype=np.int64)
    for gate_bit, position in enumerate(target_bits):
        gate_value = (np.arange(1 << num_targets, dtype=np.int64) >> (num_targets - 1 - gate_bit)) & 1
        offsets |= gate_value << position
    return offsets


def embedded_gate(gate_matrix: np.ndarray, num_qubits: int,
                  target_bits: list[int], control_bits: list[int]) -> np.ndarray:
    """Expand a small gate into the full 2**num_qubits operator it acts as.

    `target_bits` and `control_bits` are bit positions counted from the least
    significant bit. The gate only touches basis states where every control is
    1; every other basis state is left alone. Only used for 3+ target gates.
    """
    dimension = 1 << num_qubits
    basis = np.arange(dimension, dtype=np.int64)
    offsets = gate_offsets(target_bits)

    # Basis states where all controls are 1 and all targets are 0.
    control_mask = bit_mask(control_bits)
    target_mask = bit_mask(target_bits)
    usable = ((basis & control_mask) == control_mask) & ((basis & target_mask) == 0)
    anchors = basis[usable]

    full_matrix = np.eye(dimension, dtype=complex)
    for row in range(len(offsets)):
        for column in range(len(offsets)):
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


class CircuitGate:
    """A whole circuit usable as one gate: a program built on its own register.

    `CircuitGate.build(name, width, builder)` runs `builder(qc, qubits)` on a
    scratch register of `width` qubits and keeps whatever it recorded. Applying
    the result replays that program onto the parent's target qubits, so
    `qc.apply(qft_gate(4), qc[0:4])` works exactly like running the QFT circuit
    by hand -- and `op_hist` still holds the individual gates, so `uncompute`
    and `to_openqasm3` see the real operations rather than an opaque block.
    """

    def __init__(self, name: str, width: int, program: list[Gate],
                 params: list[float] | None = None):
        self.name = name
        self.width = width
        self.program = list(program)
        self.params = list(params) if params else []

    def __repr__(self):
        return f"<circuit {self.name}: {len(self.program)} gate(s) on {self.width} qubit(s)>"

    @classmethod
    def build(cls, name: str, width: int, builder, params: list[float] | None = None) -> "CircuitGate":
        """Record `builder(qc, qubits)` on a scratch register and keep the program."""
        scratch = QuantumComputer(width, name=name)
        builder(scratch, scratch.qubits)
        return cls(name, width, scratch.op_hist, params)

    def inverse(self) -> "CircuitGate":
        """The same circuit backwards: every gate inverted, in reverse order."""
        return CircuitGate(
            self.name,
            self.width,
            [gate.inverse() for gate in reversed(self.program)],
            self.params,
        )

    def __call__(self, qc: "QuantumComputer", targets: list[int], controls: list[int]):
        targets = [qc.index_of(target) for target in targets]
        if len(targets) != self.width:
            raise ValueError(
                f"'{self.name}' acts on {self.width} qubit(s), got {len(targets)} target(s)."
            )
        extra = [qc.index_of(control) for control in controls]
        qc._apply_program([gate.remap(targets, extra) for gate in self.program])


@njit(parallel=True)
def _apply_gate_kernel(state, new_state, matrix, target_mask: int,
                       control_mask: int, offsets, num_values: int):
    """Read every amplitude once, write every amplitude once.

    `state` and `new_state` are the same length; `matrix` is the gate acting on
    the targets, laid out row-major over `num_values` = 2**targets entries.

    Basis states are split three ways: those failing a control are copied
    through, those setting a target bit are left for their anchor (the same
    state with the target bits cleared, which owns the whole block), and the
    anchors read their block, multiply it by the gate, and write it back.
    """
    for basis in prange(state.shape[0]):
        if (basis & control_mask) != control_mask:
            new_state[basis] = state[basis]
        elif (basis & target_mask) == 0:
            for row in range(num_values):
                amplitude = matrix[row * num_values] * state[basis | offsets[0]]
                for column in range(1, num_values):
                    amplitude += matrix[row * num_values + column] * state[basis | offsets[column]]
                new_state[basis | offsets[row]] = amplitude


def apply_local_gate(state: np.ndarray, gate_matrix: np.ndarray, num_qubits: int,
                     target_bits: list[int], control_bits: list[int],
                     out: np.ndarray | None = None) -> np.ndarray:
    """Apply a gate to one or two targets without building a 2**n matrix.

    As everywhere else in the simulator, `gate_matrix` acts on the targets
    only and the controls say where that action is conditional. `num_qubits`
    describes the register `state` belongs to.

    The work is a single fused pass over the amplitudes, done in parallel by
    `_apply_gate_kernel`, so cost is O(2**num_qubits) instead of the
    O(4**num_qubits) of `embedded_gate` -- which is what makes wide registers
    like Shor's practical at all.

    `state` is never written to. Pass `out=` to choose where the result lands;
    reusing one buffer across gates is what keeps the cost near a plain copy,
    since a fresh 32 MB array would fault in new pages every gate.
    """
    if state.size != 1 << num_qubits:
        raise ValueError(f"A {num_qubits}-qubit register holds {1 << num_qubits} amplitudes, not {state.size}.")

    flat = state.reshape(-1)
    if out is None:
        new_state = np.empty_like(flat)
    else:
        new_state = out.reshape(-1)
        if new_state.size != flat.size:
            raise ValueError(f"`out` holds {new_state.size} amplitudes, expected {flat.size}.")
        if np.shares_memory(new_state, flat):
            raise ValueError("`out` must not overlap `state`; the gate needs both at once.")

    _apply_gate_kernel(
        flat,
        new_state,
        np.asarray(gate_matrix, dtype=complex).ravel(),
        bit_mask(target_bits),
        bit_mask(control_bits),
        gate_offsets(target_bits),
        1 << len(target_bits),
    )
    return new_state.reshape(-1, 1)


def _is_diagonal_local(gate: Gate) -> bool:
    """True for a single-target gate with nothing off the diagonal (a phase)."""
    return (
        len(gate.targets) == 1
        and gate.matrix.shape == (2, 2)
        and gate.matrix[0, 1] == 0
        and gate.matrix[1, 0] == 0
    )


def _is_swap_local(gate: Gate) -> bool:
    """True for a plain SWAP: two targets, no controls, the swap matrix."""
    return (
        len(gate.targets) == 2
        and not gate.controls
        and gate.matrix.shape == (4, 4)
        and np.array_equal(gate.matrix, gm.swap_MAT)
    )


@njit(parallel=True)
def _fused_diagonal_kernel(new_state, state, control_masks, target_bits, diag0, diag1):
    """Apply a whole run of diagonal single-target gates in one pass.

    Each gate is `diag(a, b)` on its own target with its own controls, so the
    combined action on a basis state is one complex product computed inline.
    Fusing a run this way replaces one read+write of the state per gate with a
    single pass per run, which is what makes the QFT's t(t+1)/2 controlled
    phases cost t passes instead of t(t+1)/2.
    """
    num_gates = control_masks.shape[0]
    for basis in prange(state.shape[0]):
        value = 1.0 + 0.0j
        for gate_index in range(num_gates):
            if (basis & control_masks[gate_index]) == control_masks[gate_index]:
                if (basis >> target_bits[gate_index]) & 1:
                    value *= diag1[gate_index]
                else:
                    value *= diag0[gate_index]
        new_state[basis] = state[basis] * value


def _fused_diagonal_numpy(new_state, state, control_masks, target_bits, diag0, diag1):
    """NumPy stand-in for `_fused_diagonal_kernel` when numba is unavailable."""
    basis = np.arange(state.shape[0], dtype=np.int64)
    value = np.ones(state.shape[0], dtype=complex)
    for gate_index in range(len(control_masks)):
        allowed = (basis & control_masks[gate_index]) == control_masks[gate_index]
        bit = (basis >> target_bits[gate_index]) & 1
        value *= np.where(
            allowed,
            np.where(bit == 1, diag1[gate_index], diag0[gate_index]),
            1.0 + 0.0j,
        )
    new_state[:] = state * value


@njit(parallel=True)
def _swap_run_kernel(new_state, state, pairs):
    """Apply a run of disjoint SWAPs as one permutation pass over the state.

    `pairs` holds the two bit positions each swap exchanges. Disjoint swaps
    compose into a single involution, so every amplitude moves exactly once:
    this is the whole bit-reversal layer of a QFT in one read + one write
    instead of n/2 gate applications.
    """
    num_pairs = pairs.shape[0]
    for basis in prange(state.shape[0]):
        # numba's parfor pass mis-infers the prange index as float64 once it
        # is carried through a loop-reassigned alias; the cast keeps it int.
        swapped = np.int64(basis)
        for pair_index in range(num_pairs):
            low = pairs[pair_index, 0]
            high = pairs[pair_index, 1]
            bit_low = (swapped >> low) & 1
            bit_high = (swapped >> high) & 1
            swapped = (swapped & ~(1 << low) & ~(1 << high)) | (bit_low << high) | (bit_high << low)
        new_state[basis] = state[swapped]


def _swap_run_numpy(new_state, state, pairs):
    """NumPy stand-in for `_swap_run_kernel` when numba is unavailable."""
    swapped = np.arange(state.shape[0], dtype=np.int64)
    for pair_index in range(pairs.shape[0]):
        low = pairs[pair_index, 0]
        high = pairs[pair_index, 1]
        bit_low = (swapped >> low) & 1
        bit_high = (swapped >> high) & 1
        swapped = (swapped & ~(1 << low) & ~(1 << high)) | (bit_low << high) | (bit_high << low)
    new_state[:] = state[swapped]


# ============================================================
# Fast QFT/IQFT using dense matrices for small registers
# ============================================================

def _qft_dense_matrix(num_targets: int, inverse: bool = False) -> np.ndarray:
    """Return the dense QFT or IQFT matrix for the given number of targets."""
    from quantum.gate_matrix import qft_MAT, iqft_MAT
    return iqft_MAT(num_targets) if inverse else qft_MAT(num_targets)


def _qft_dense_apply(state, out, qubit_bits: np.ndarray, inverse: bool):
    """Apply QFT/IQFT using dense matrix multiplication for small registers.
    
    Assumes the target qubits are the entire register (no spectators).
    """
    num_targets = len(qubit_bits)
    if num_targets == 0:
        return

    # Get dense matrix
    matrix = _qft_dense_matrix(len(qubit_bits), inverse=inverse)

    # Apply matrix to the state vector
    out[:] = (matrix @ state.reshape(-1, 1)).flatten()


def _qft_numpy(state, new_state, qubit_bits: np.ndarray, inverse: bool):
    """NumPy stand-in for `_qft_kernel` when numba is unavailable."""
    num_targets = len(qubit_bits)
    if num_targets == 0:
        new_state[:] = state[:]
        return

    # Use dense matrix for small registers
    if num_targets <= 10 and num_targets == state.shape[0].bit_length() - 1:
        # Allocate output buffer
        result = np.empty_like(state)
        _qft_dense_apply(state, result, qubit_bits, inverse)
        new_state[:] = result
        return

    # Fallback: use circuit approach (not implemented in numpy fallback)
    raise NotImplementedError("Numba required for registers > 10 qubits")


# ============================================================

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
            # Look up custom operator in instance registry first, then global
            op, kind = get_global_operator(operation)
            if op is None:
                op = self.get_operator(operation)
                kind = getattr(op, '_quantar_kind', 'matrix') if op else 'matrix'
            else:
                kind = kind or getattr(op, '_quantar_kind', 'matrix')
            if op is None:
                raise ValueError(f"Unknown operator: '{operation}'")
            
            if kind == "matrix":
                if callable(op):
                    matrix = op(self) if getattr(op, '_quantar_requires_instance', False) else op()
                else:
                    matrix = op
                if not isinstance(matrix, np.ndarray):
                    raise ValueError(f"Operator '{operation}' must return a numpy array")
                if target_qubits is None:
                    raise ValueError("Custom operator requires target qubits")
                self.apply_gate(matrix, target_qubits, control_qubits, qasm_name=operation)
                return
            
            elif kind == "unary":
                if target_qubits is None or len(target_qubits) != 1:
                    raise ValueError(f"Unary operator '{operation}' requires exactly one target qubit")
                target = self.index_of(target_qubits[0])
                if callable(op):
                    if getattr(op, '_quantar_requires_instance', False):
                        result = op(self, target)
                    else:
                        result = op(target)
                return result
            
            elif kind == "binary":
                if target_qubits is None or len(target_qubits) != 2:
                    raise ValueError(f"Binary operator '{operation}' requires exactly two target qubits (control, target)")
                control = self.index_of(target_qubits[0])
                target = self.index_of(target_qubits[1])
                if callable(op):
                    if getattr(op, '_quantar_requires_instance', False):
                        op(self, control, target)
                    else:
                        op(control, target)
                return
            
            elif kind == "callable":
                if target_qubits is None:
                    raise ValueError("Callable operator requires target qubits")
                targets = [self.index_of(q) for q in target_qubits]
                controls = [self.index_of(q) for q in (control_qubits or [])]
                if callable(op):
                    if getattr(op, '_quantar_requires_instance', False):
                        op(self, targets, controls)
                    else:
                        op(targets, controls)
                return
            
            # Default: treat as matrix
            if callable(op):
                matrix = op(self) if getattr(op, '_quantar_requires_instance', False) else op()
            else:
                matrix = op
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"Operator '{operation}' must return a numpy array")
            if target_qubits is None:
                raise ValueError("Custom operator requires target qubits")
            self.apply_gate(matrix, target_qubits, control_qubits, qasm_name=operation)
            return

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
            _swap_run_kernel(out, flat, pairs)
        else:
            _swap_run_numpy(out, flat, pairs)

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
            _fused_diagonal_kernel(out, flat, control_masks, target_bits, diag0, diag1)
        else:
            _fused_diagonal_numpy(out, flat, control_masks, target_bits, diag0, diag1)

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

        For registers ≤ 10 qubits, uses a dense matrix multiplication
        (pre-computed QFT/IQFT matrix from gate_matrix) which is faster.
        For larger registers, falls back to the optimized circuit approach
        with fused diagonal runs.

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

        num_targets = len(targets)
        if num_targets <= 10 and num_targets == self.num_qubits:
            # Fast path: dense matrix multiplication (only when targets = all qubits)
            qubit_bits = np.array([self.bit_position(t) for t in targets], dtype=np.int64)

            if self._scratch is None or self._scratch.shape != self.state_vector.shape:
                self._scratch = np.empty_like(self.state_vector)
            out = self._scratch.reshape(-1)

            _qft_dense_apply(self.state_vector.reshape(-1), out, qubit_bits, inverse)

            previous_state = self.state_vector
            self.state_vector = out.reshape(-1, 1)
            self._scratch = previous_state
        else:
            # Larger register or partial register: fall back to optimized circuit approach
            from stl.quantum_algs.shors_alg import qft_program, iqft_program
            program = iqft_program(self, self.qubits[:num_targets]) if inverse else qft_program(self, self.qubits[:num_targets])
            self.apply(program)

        rec = self._recording if record is None else record
        if rec:
            self.op_hist.append(Gate(
                np.eye(1, dtype=complex),  # placeholder
                [], [],
                name="iqft" if inverse else "qft",
                params=[float(inverse)],
            ))

        return self

    def iqft(self, qubits: list[Qubit | int | str], **kwargs) -> "QuantumComputer":
        """Apply a fast Inverse Quantum Fourier Transform.

        Equivalent to `qft(qubits, inverse=True)`.
        """
        return self.qft(qubits, inverse=True, **kwargs)

    def measure_pauli(self, pauli: str, qubits: list[Qubit | int | str]) -> float:
        """Measure the expectation value of a Pauli string in the Z basis.

        Args:
            pauli: Pauli string like 'X', 'Y', 'Z', 'XX', 'ZZ', 'XY', 'XYZ', etc.
            qubits: List of qubits to measure (must match length of pauli string)

        Returns:
            Expectation value <psi|pauli|psi> as a float.
        """
        targets = [self.index_of(q) for q in qubits]
        if len(targets) != len(pauli):
            raise ValueError(f"Pauli string length ({len(pauli)}) must match number of qubits ({len(targets)})")

        # Apply basis rotation before measurement
        for idx, p in zip(targets, pauli):
            if p == 'X':
                self.apply_gate(gm.h_MAT, [self.qubits[idx]])
            elif p == 'Y':
                # Y basis: S† then H
                self.apply_gate(gm.sdg_MAT, [self.qubits[idx]])
                self.apply_gate(gm.h_MAT, [self.qubits[idx]])
            elif p == 'Z':
                pass  # Z basis is computational basis
            elif p == 'I':
                pass  # Identity - no rotation needed
            else:
                raise ValueError(f"Unknown Pauli operator: {p}")

        # Measure all target qubits in Z basis
        parity_values = []
        for _ in range(1000):  # Fixed shots for now
            # Simulate measurement by sampling from probability distribution
            probs = np.abs(self.state_vector.flatten()) ** 2
            probs /= probs.sum()
            outcome = rnd.choice(self.dimension, p=probs)
            
            # Compute parity of measured bits
            parity = 1.0
            for idx in targets:
                bit = (outcome >> (self.num_qubits - 1 - idx)) & 1
                if bit == 1:
                    parity *= -1
            parity_values.append(parity)

        # Clean up: apply inverse rotations to restore state
        for idx, p in zip(targets, pauli):
            if p == 'X':
                self.apply_gate(gm.h_MAT, [self.qubits[idx]])
            elif p == 'Y':
                self.apply_gate(gm.s_MAT, [self.qubits[idx]])
                self.apply_gate(gm.h_MAT, [self.qubits[idx]])
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
