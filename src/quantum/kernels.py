"""Numba kernels and their numpy fallbacks -- the simulator's performance core.

Every pass over the amplitudes funnels through one of these:

    apply_gate_kernel      one gate (1-2 targets), with any controls
    fused_diagonal_kernel  a whole run of diagonal phase gates in one pass
    swap_run_kernel        a whole run of disjoint SWAPs as one permutation
    sample_counts          draw measurement outcomes from probabilities

Each `*_kernel` dispatches to a `*_numpy` twin for small registers or when numba
is unavailable (`HAVE_NUMBA` is False), so the simulator stays correct with or
without the JIT. Numba is imported lazily on the first large-register run; small
programs avoid its several-second compile cost.

Quote of the house rules these obey: read each amplitude once, write it once;
never mutate the input state in place (the caller passes an `out` buffer).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

_NUMBA_REQUESTED = os.environ.get("QUANTAR_NUMBA", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
try:
    _NUMBA_INSTALLED = importlib.util.find_spec("numba") is not None
except (ImportError, ValueError):
    _NUMBA_INSTALLED = False

# The JIT is reserved for registers where its warm speed pays back the
# several-second compile cost. Small programs use the vectorized NumPy path.
HAVE_NUMBA = _NUMBA_REQUESTED and _NUMBA_INSTALLED
NUMBA_MIN_AMPLITUDES = int(
    os.environ.get("QUANTAR_NUMBA_MIN_AMPLITUDES", str(1 << 20))
)

prange = range
_compiled_kernels: dict[str, object] = {}


def should_use_numba(num_amplitudes: int) -> bool:
    """Return whether this register is large enough to benefit from the JIT."""
    return HAVE_NUMBA and num_amplitudes >= NUMBA_MIN_AMPLITUDES


def _compiled(name: str, implementation):
    """Compile a kernel on first use, returning None if numba is unavailable."""
    if name in _compiled_kernels:
        return _compiled_kernels[name]

    try:
        from numba import jit, prange as numba_prange
    except Exception:
        global HAVE_NUMBA
        HAVE_NUMBA = False
        _compiled_kernels[name] = None
        return None

    global prange
    prange = numba_prange
    compiled = jit(parallel=True)(implementation)
    _compiled_kernels[name] = compiled
    return compiled


def _apply_gate_kernel_impl(state, new_state, matrix, target_mask: int,
                            control_mask: int, offsets, num_values: int):
    """Apply one gate to the amplitudes of `state`, writing into `new_state`.

    `state` and `new_state` have the same length; `matrix` is the gate acting on
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


def _apply_gate_kernel_numpy(state, new_state, matrix, target_mask: int,
                             control_mask: int, offsets, num_values: int):
    """Vectorized stand-in for `_apply_gate_kernel_impl`."""
    basis = np.arange(state.shape[0], dtype=np.int64)
    new_state[:] = state

    if control_mask:
        allowed = (basis & control_mask) == control_mask
    else:
        allowed = np.ones(state.shape[0], dtype=bool)
    anchors = allowed & ((basis & target_mask) == 0)
    if not np.any(anchors):
        return

    base = basis[anchors]
    indices = base[:, None] | np.asarray(offsets, dtype=np.int64)[None, :]
    gate = np.asarray(matrix).reshape(num_values, num_values)
    new_state[indices] = state[indices] @ gate.T


def apply_gate_kernel(state, new_state, matrix, target_mask: int,
                      control_mask: int, offsets, num_values: int):
    """Apply one gate through the JIT for large registers, NumPy otherwise."""
    if should_use_numba(state.shape[0]):
        compiled = _compiled("apply_gate", _apply_gate_kernel_impl)
        if compiled is not None:
            compiled(
                state, new_state, matrix, target_mask, control_mask,
                offsets, num_values,
            )
            return
    _apply_gate_kernel_numpy(
        state, new_state, matrix, target_mask, control_mask, offsets,
        num_values,
    )


def _fused_diagonal_kernel_impl(new_state, state, control_masks, target_bits, diag0, diag1):
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


def fused_diagonal_numpy(new_state, state, control_masks, target_bits, diag0, diag1):
    """NumPy stand-in for `fused_diagonal_kernel` when numba is unavailable."""
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


def fused_diagonal_kernel(new_state, state, control_masks, target_bits, diag0, diag1):
    """Fuse diagonal gates through the JIT for large registers, NumPy otherwise."""
    if should_use_numba(state.shape[0]):
        compiled = _compiled("fused_diagonal", _fused_diagonal_kernel_impl)
        if compiled is not None:
            compiled(new_state, state, control_masks, target_bits, diag0, diag1)
            return
    fused_diagonal_numpy(
        new_state, state, control_masks, target_bits, diag0, diag1,
    )


def _swap_run_kernel_impl(new_state, state, pairs):
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


def swap_run_numpy(new_state, state, pairs):
    """NumPy stand-in for `swap_run_kernel` when numba is unavailable."""
    swapped = np.arange(state.shape[0], dtype=np.int64)
    for pair_index in range(pairs.shape[0]):
        low = pairs[pair_index, 0]
        high = pairs[pair_index, 1]
        bit_low = (swapped >> low) & 1
        bit_high = (swapped >> high) & 1
        swapped = (swapped & ~(1 << low) & ~(1 << high)) | (bit_low << high) | (bit_high << low)
    new_state[:] = state[swapped]


def swap_run_kernel(new_state, state, pairs):
    """Run disjoint SWAPs through the JIT for large registers, NumPy otherwise."""
    if should_use_numba(state.shape[0]):
        compiled = _compiled("swap_run", _swap_run_kernel_impl)
        if compiled is not None:
            compiled(new_state, state, pairs)
            return
    swap_run_numpy(new_state, state, pairs)


def sample_counts(probabilities: np.ndarray, num_samples: int, dimension: int) -> np.ndarray:
    """Draw `num_samples` basis states from `probabilities`; return counts per state.

    Vectorized rather than a numba loop: `searchsorted` over the cumulative
    distribution does the inverse-CDF sampling in C, and it avoids the data race
    a parallel `counts[draw] += 1` would introduce.
    """
    cumulative = np.cumsum(probabilities)
    if cumulative[-1] > 0:
        cumulative /= cumulative[-1]
    cumulative[-1] = 1.0
    draws = np.searchsorted(cumulative, np.random.rand(num_samples))
    return np.bincount(draws, minlength=dimension)
