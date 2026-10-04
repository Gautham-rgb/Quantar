from collections.abc import Callable
from numpy import asin, floor, pi, sin, sqrt
import quantum.qubit_abs as qa
import quantum.gate_matrix as gm

def optimal_iterations(num_items: int, num_targets: int) -> int:
    if num_items <= 0 or num_targets <= 0:
        raise ValueError("num_items and num_targets must both be positive.")
    num_targets = min(num_targets, num_items)
    theta = asin(sqrt(num_targets / num_items))
    max_iterations = int(floor(pi / (4 * theta))) + 2
    return int(max(range(max_iterations + 1), key=lambda k: sin((2 * k + 1) * theta) ** 2))

def grover_oracle(qc: qa.QuantumComputer, data_qubits: list, ancilla: qa.Qubit, logic_func: Callable):
    if not data_qubits:
        return
    qc.apply_gate(gm.px_MAT, [ancilla])
    qc.apply_gate(gm.h_MAT, [ancilla])
    logic_func(qc, data_qubits, ancilla)
    qc.apply_gate(gm.h_MAT, [ancilla])
    qc.apply_gate(gm.px_MAT, [ancilla])

def grover_diffuser(qc: qa.QuantumComputer, data_qubits: list):
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
    for qubit in data_qubits:
        qc.apply_gate(gm.h_MAT, [qubit])

def grovers_alg(
        qc: qa.QuantumComputer,
        logic_func: Callable,
        num_targets: int = 1,
        scratch_qubits: list | None = None,
    ) -> int:
    
    ancilla_qubit = qc[-1]
    scratch_indices = set()
    if scratch_qubits:
        for qubit in scratch_qubits:
            scratch_indices.add(qubit.index if isinstance(qubit, qa.Qubit) else qc[qubit].index)
    scratch_indices.discard(ancilla_qubit.index)
    data_qubits = [
        qubit for qubit in qc.qubits
        if qubit.index != ancilla_qubit.index and qubit.index not in scratch_indices
    ]
    if not data_qubits:
        raise ValueError("Grover's algorithm needs at least one data qubit (plus an ancilla).")
    for qubit in data_qubits:
        qc.apply_gate(gm.h_MAT, [qubit])
    iterations = optimal_iterations(2 ** len(data_qubits), num_targets)
    for _ in range(iterations):
        grover_oracle(qc, data_qubits, ancilla_qubit, logic_func) #type: ignore
        grover_diffuser(qc, data_qubits)
    return iterations

if __name__ == "__main__":
    def run_simulation(qc):
        def problem_constraints(qc: qa.QuantumComputer, data_qubits: list, ancilla_qubit: qa.Qubit):
            qc.apply_gate(gm.px_MAT, [ancilla_qubit], control_qubits=data_qubits)
        grovers_alg(qc, problem_constraints)
    quantum_computer = qa.QuantumComputer(4)
    counts = quantum_computer.sample(run_simulation, 1000)
    results: dict[str, int] = {}
    for bits, count in counts.items():
        data_bits = bits[:3]
        results[data_bits] = results.get(data_bits, 0) + count
    print("index : samples")
    for bits, count in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{bits} : {count}")
