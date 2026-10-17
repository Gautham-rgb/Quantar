"""Numba kernels and their numpy fallbacks -- the simulator's performance core.

Every pass over the amplitudes funnels through one of these:

    apply_gate_kernel      one gate (1-2 targets), with any controls
    fused_diagonal_kernel  a whole run of diagonal phase gates in one pass
    swap_run_kernel        a whole run of disjoint SWAPs as one permutation
    sample_counts          draw measurement outcomes from probabilities

Each `*_kernel` has a `*_numpy` twin that runs when numba is unavailable
(`HAVE_NUMBA` is False), so the simulator stays correct with or without the JIT.

Quote of the house rules these obey: read each amplitude once, write it once;
never mutate the input state in place (the caller passes an `out` buffer).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import jit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def jit(*args, **kwargs):
        def decorate(func):
            return func
        return decorate

    prange = range


@jit(parallel=True)
def apply_gate_kernel(state, new_state, matrix, target_mask: int,
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


@jit(parallel=True)
def fused_diagonal_kernel(new_state, state, control_masks, target_bits, diag0, diag1):
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


@jit(parallel=True)
def swap_run_kernel(new_state, state, pairs):
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
