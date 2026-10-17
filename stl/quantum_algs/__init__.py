"""Quantar's quantum algorithms.

Everything is importable from the package root, so callers can write::

    from stl.quantum_algs import shors_alg, grovers_alg
    from stl.quantum_algs import QuantumOptimizer, Ansatz, Hamiltonian
    from stl.quantum_algs import qft_gate, iqft_gate, factor

Modules:
    shors_alg        Shor's factoring (incl. QFT/IQFT programs and gates)
    grovers_alg      Grover's search (oracle + diffuser)
    quantum_optim    Ansatz, Hamiltonian, Pauli expectations
    quantum_optim_hl QuantumOptimizer and penalty helpers
    fast_factor      classical factorization utilities
"""

from stl.quantum_algs.shors_alg import (
    shors_alg,
    shors_alg_optim,
    quantum_subroutine,
    mod_exp,
    qft_program,
    iqft_program,
    qft_circuit,
    iqft_circuit,
    qft_gate,
    iqft_gate,
)
from stl.quantum_algs.grovers_alg import (
    grovers_alg,
    grover_oracle,
    grover_diffuser,
    optimal_iterations,
)
from stl.quantum_algs.quantum_optim import (
    Ansatz,
    Hamiltonian,
    hardware_efficient_ansatz,
    measure_pauli_expectation,
    pauli_string_to_tensor,
)
from stl.quantum_algs.quantum_optim_hl import (
    QuantumOptimizer,
    OptimizeResult,
    Penalty,
    penalized_cost,
    minimize_cost,
)
from stl.quantum_algs.fast_factor import (
    factor,
    factor_pair,
    is_prime,
    factorize,
)

__all__ = [
    # Shor's
    "shors_alg", "shors_alg_optim", "quantum_subroutine", "mod_exp",
    "qft_program", "iqft_program", "qft_circuit", "iqft_circuit",
    "qft_gate", "iqft_gate",
    # Grover's
    "grovers_alg", "grover_oracle", "grover_diffuser", "optimal_iterations",
    # Optimization
    "Ansatz", "Hamiltonian", "hardware_efficient_ansatz",
    "measure_pauli_expectation", "pauli_string_to_tensor",
    "QuantumOptimizer", "OptimizeResult", "Penalty",
    "penalized_cost", "minimize_cost",
    # Classical helpers
    "factor", "factor_pair", "is_prime", "factorize",
]
