"""Grover's search.

The oracle is built from compute / kickback / uncompute, which is why the
simulator keeps an operation history: `uncompute` is what returns the ancilla
to |0> so it stays clean for the next iteration.
"""

from collections.abc import Callable

from numpy import asin, floor, pi, sin, sqrt

import quantum.gate_matrix as gm
import quantum.qubit_abs as qa


def optimal_iterations(num_items: int, num_targets: int) -> int:
    """How many Grover iterations give the highest success probability."""
    if num_items <= 0 or num_targets <= 0:
        raise ValueError("num_items and num_targets must both be positive.")

    num_targets = min(num_targets, num_items)
    theta = asin(sqrt(num_targets / num_items))
    max_iterations = int(floor(pi / (4 * theta))) + 2

    def success_probability(iteration: int) -> float:
        return sin((2 * iteration + 1) * theta) ** 2

    return int(max(range(max_iterations + 1), key=success_probability))


def grover_oracle(qc: qa.QuantumComputer, data_qubits: list, ancilla_qubit: qa.Qubit, logic_func: Callable):
    """Flip the ancilla exactly where `logic_func` says a target sits.

    `logic_func(qc, data_qubits, ancilla_qubit)` marks a target by X-ing the
    ancilla controlled on every data qubit. The gates around it leave the
    ancilla back at |0>, so the oracle can be reused.
    """
    if not data_qubits:
        return

    qc.apply_gate(gm.px_MAT, [ancilla_qubit])
    qc.apply_gate(gm.h_MAT, [ancilla_qubit])
    logic_func(qc, data_qubits, ancilla_qubit)
    qc.apply_gate(gm.h_MAT, [ancilla_qubit])
    qc.apply_gate(gm.px_MAT, [ancilla_qubit])


def grover_diffuser(qc: qa.QuantumComputer, data_qubits: list):
    """Reflect the amplitudes about their mean: H X (multi-)CZ X H."""
    if not data_qubits:
        return

    for qubit in data_qubits:
        qc.apply_gate(gm.h_MAT, [qubit])
        qc.apply_gate(gm.px_MAT, [qubit])

    if len(data_qubits) > 1:
        qc.apply_gate(gm.pz_MAT, [data_qubits[0]], control_qubits=data_qubits[1:])
    else:
        qc.apply_gate(gm.pz_MAT, [data_qubits[0]])

    for qubit in data_qubits:
        qc.apply_gate(gm.px_MAT, [qubit])
        qc.apply_gate(gm.h_MAT, [qubit])


def grovers_alg(
    qc: qa.QuantumComputer,
    logic_func: Callable,
    num_targets: int = 1,
    scratch_qubits: list | None = None,
) -> int:
    """Run Grover's algorithm and return the number of iterations used.

    The last qubit is the oracle ancilla. `scratch_qubits` are held out of the
    search, so every remaining qubit becomes a data qubit.
    """
    ancilla_qubit = qc[-1]

    excluded_indices = {ancilla_qubit.index}
    for qubit in scratch_qubits or []:
        excluded_indices.add(qubit.index if isinstance(qubit, qa.Qubit) else qc[qubit].index)

    data_qubits = [qubit for qubit in qc.qubits if qubit.index not in excluded_indices]
    if not data_qubits:
        raise ValueError("Grover's algorithm needs at least one data qubit (plus an ancilla).")

    for qubit in data_qubits:
        qc.apply_gate(gm.h_MAT, [qubit])

    iterations = optimal_iterations(2 ** len(data_qubits), num_targets)
    for _ in range(iterations):
        grover_oracle(qc, data_qubits, ancilla_qubit, logic_func)
        grover_diffuser(qc, data_qubits)

    return iterations


def run_simulation(qc: qa.QuantumComputer):
    """Demo problem: mark the all-ones data register."""
    def mark_all_ones(qc: qa.QuantumComputer, data_qubits: list, ancilla_qubit: qa.Qubit):
        qc.apply_gate(gm.px_MAT, [ancilla_qubit], control_qubits=data_qubits)

    grovers_alg(qc, mark_all_ones)


if __name__ == "__main__":
    quantum_computer = qa.QuantumComputer(4)
    counts = quantum_computer.sample(run_simulation, 1000)

    results: dict[str, int] = {}
    for bits, count in counts.items():
        data_bits = bits[:3]
        results[data_bits] = results.get(data_bits, 0) + count

    print("index : samples")
    for bits, count in sorted(results.items(), key=lambda item: -item[1]):
        print(f"{bits} : {count}")
