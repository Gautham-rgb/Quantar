"""quantum -- a small, readable state-vector quantum simulator.

Public API::

    from quantum import QuantumComputer, Qubit, Gate, CircuitGate
    from quantum import h_MAT, px_MAT, crk, to_openqasm3

Internals are grouped by concern (seven files, not a pile of fragments):

    qubit_abs     QuantumComputer and Qubit -- the simulator itself
    gate_abs      Gate, CompositeGate, CircuitGate -- how operations are recorded
    gate_matrix   gate matrices and their OpenQASM 3 names
    kernels       numba kernels + numpy fallbacks (the hot loops)
    basic_funcs   bit helpers and the apply entry points wiring kernels to gates
    composites    crk (controlled phase shifts)
    qasm          OpenQASM 3 export

QFT/IQFT circuit builders (`qft_program`, `qft_gate`, ...) live in
`stl.quantum_algs.shors_alg`, next to the algorithm that uses them.
"""

from quantum.qubit_abs import QuantumComputer, Qubit
from quantum.gate_abs import Gate, CompositeGate, CircuitGate
from quantum import gate_matrix
from quantum import kernels
from quantum.gate_matrix import (
    h_MAT, px_MAT, py_MAT, pz_MAT, i_MAT,
    s_MAT, sdg_MAT, t_MAT, tdg_MAT,
    swap_MAT, iswap_MAT, swap_sqrt_MAT,
    rx_MAT, ry_MAT, rz_MAT, p_MAT, cp_MAT, rk_MAT,
    qft_MAT, iqft_MAT,
    named_gate, name_is_controlled,
)
from quantum.kernels import (
    apply_gate_kernel, fused_diagonal_kernel, swap_run_kernel, sample_counts,
    HAVE_NUMBA,
)
from quantum.composites import crk
from quantum.qasm import to_openqasm3

__all__ = [
    "QuantumComputer",
    "Qubit",
    "Gate",
    "CompositeGate",
    "CircuitGate",
    "gate_matrix",
    "kernels",
    "h_MAT", "px_MAT", "py_MAT", "pz_MAT", "i_MAT",
    "s_MAT", "sdg_MAT", "t_MAT", "tdg_MAT",
    "swap_MAT", "iswap_MAT", "swap_sqrt_MAT",
    "rx_MAT", "ry_MAT", "rz_MAT", "p_MAT", "cp_MAT", "rk_MAT",
    "qft_MAT", "iqft_MAT",
    "named_gate", "name_is_controlled",
    "apply_gate_kernel", "fused_diagonal_kernel", "swap_run_kernel",
    "sample_counts", "HAVE_NUMBA",
    "crk",
    "to_openqasm3",
]
