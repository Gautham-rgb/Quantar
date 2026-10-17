"""Amplitude plumbing: bit helpers and the gate-application entry points.

The kernels themselves live in `quantum.kernels`; this module wires them to the
simulator's conventions:

    apply_local_gate          the main path: one kernel pass for a 1-2 target gate
    _apply_gate_tensor        same, used as the >26-qubit fallback for einsum
    _apply_gate_tensor_einsum optional tensor-network path for small registers

Plus the small bit arithmetic the kernels need (`bit_mask`, `gate_offsets`,
`embedded_gate`) and the predicates that let `QuantumComputer` batch runs.
"""

import numpy as np
from .gate_abs import Gate
from . import gate_matrix as gm
from quantum.kernels import apply_gate_kernel


def _apply_gate_tensor(state: np.ndarray, gate_matrix: np.ndarray, num_qubits: int,
                       target_bits: list[int], control_bits: list[int],
                       out: np.ndarray | None = None) -> np.ndarray:
    """Apply a gate with the numba kernel (fast and correct for every size).

    The tensor/einsum alternative was slower for larger registers, so the kernel
    is what the simulator reaches for by default.
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

    apply_gate_kernel(
        flat, new_state,
        np.asarray(gate_matrix, dtype=complex).ravel(),
        bit_mask(target_bits),
        bit_mask(control_bits),
        gate_offsets(target_bits),
        1 << len(target_bits),
    )
    return new_state.reshape(-1, 1)


def _apply_gate_tensor_einsum(state: np.ndarray, gate_matrix: np.ndarray, num_qubits: int,
                              target_bits: list[int], control_bits: list[int],
                              out: np.ndarray | None = None) -> np.ndarray:
    """Apply a 1-2 target gate via einsum on the reshaped (2,)*n amplitude tensor.

    Kept as an alternative to the kernel for small registers and as the natural
    entry point for tensor-network style work. Qubit convention: q0 is the MSB,
    so bit position = num_qubits - 1 - qubit_index and axis 0 of the tensor is
    q0. Falls back to the kernel for >2 targets or >26 qubits (einsum subscripts
    are single characters).
    """
    if state.size != 1 << num_qubits:
        raise ValueError(f"A {num_qubits}-qubit register holds {1 << num_qubits} amplitudes, not {state.size}.")

    if num_qubits > 26 or len(target_bits) > 2:
        return _apply_gate_tensor(state, gate_matrix, num_qubits, target_bits, control_bits, out)

    tensor = state.reshape((2,) * num_qubits)

    if out is None:
        new_tensor = np.empty_like(tensor)
    else:
        new_tensor = out.reshape((2,) * num_qubits)
        if new_tensor.size != tensor.size:
            raise ValueError(f"`out` holds {new_tensor.size} amplitudes, expected {tensor.size}.")
        if np.shares_memory(new_tensor, tensor):
            raise ValueError("`out` must not overlap `state`; the gate needs both at once.")

    # Keep only the amplitudes whose control bits are all 1.
    if control_bits:
        control_mask = np.ones((2,) * num_qubits, dtype=bool)
        for control_bit in control_bits:
            axis = num_qubits - 1 - control_bit
            selector = [slice(None)] * num_qubits
            selector[axis] = 1#type:ignore
            axis_mask = np.zeros((2,) * num_qubits, dtype=bool)
            axis_mask[tuple(selector)] = True
            control_mask &= axis_mask

        new_tensor[~control_mask] = tensor[~control_mask]
        tensor = tensor.copy()
        tensor[~control_mask] = 0
    else:
        control_mask = np.ones((2,) * num_qubits, dtype=bool)

    target_axes = sorted(num_qubits - 1 - bit for bit in target_bits)
    all_axes = 'abcdefghijklmnopqrstuvwxyz'[:num_qubits]

    if len(target_axes) == 1:
        axis = target_axes[0]
        gate_idx = 'Y' + all_axes[axis]
        out_indices = all_axes[:axis] + 'Y' + all_axes[axis + 1:]
        np.einsum(f'{all_axes},{gate_idx}->{out_indices}', tensor,
                  np.asarray(gate_matrix).reshape(2, 2), out=new_tensor)
    else:
        axis1, axis2 = target_axes
        idx1, idx2 = all_axes[axis1], all_axes[axis2]
        gate_idx = 'Y' + idx1 + 'Z' + idx2
        out_indices = list(all_axes)
        out_indices[axis1] = 'Y'
        out_indices[axis2] = 'Z'
        np.einsum(f'{all_axes},{gate_idx}->{"".join(out_indices)}', tensor,
                  np.asarray(gate_matrix).reshape(2, 2, 2, 2), out=new_tensor)

    if control_bits:
        new_tensor[~control_mask] = state.reshape((2,) * num_qubits)[~control_mask]

    return new_tensor.reshape(-1, 1)


def apply_local_gate(state: np.ndarray, gate_matrix: np.ndarray, num_qubits: int,
                     target_bits: list[int], control_bits: list[int],
                     out: np.ndarray | None = None) -> np.ndarray:
    """Apply a gate to one or two targets without building a 2**n matrix.

    As everywhere else in the simulator, `gate_matrix` acts on the targets
    only and the controls say where that action is conditional. `num_qubits`
    describes the register `state` belongs to.

    The work is a single fused pass over the amplitudes, done in parallel by
    `apply_gate_kernel`, so cost is O(2**num_qubits) instead of the
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

    apply_gate_kernel(
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
        and np.array_equal(gate.matrix, gm.swap_MAT())
    )


def _qft_dense_matrix(num_targets: int, inverse: bool = False) -> np.ndarray:
    """Return the dense QFT or IQFT matrix for the given number of targets."""
    from quantum.gate_matrix import qft_MAT, iqft_MAT
    return iqft_MAT(num_targets) if inverse else qft_MAT(num_targets)


def _qft_dense_apply(state, out, qubit_bits: np.ndarray, inverse: bool):
    """Apply QFT/IQFT as one dense matrix multiply (small full-register case)."""
    if len(qubit_bits) == 0:
        return
    matrix = _qft_dense_matrix(len(qubit_bits), inverse=inverse)
    out[:] = (matrix @ state.reshape(-1, 1)).flatten()


def _qft_numpy(state, new_state, qubit_bits: np.ndarray, inverse: bool):
    """NumPy stand-in for the QFT kernel when numba is unavailable."""
    num_targets = len(qubit_bits)
    if num_targets == 0:
        new_state[:] = state[:]
        return

    if num_targets <= 10 and num_targets == state.shape[0].bit_length() - 1:
        _qft_dense_apply(state, new_state, qubit_bits, inverse)
        return

    raise NotImplementedError("Numba required for registers > 10 qubits")


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
