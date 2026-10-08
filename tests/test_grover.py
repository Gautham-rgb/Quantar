import numpy as np
from numpy import sin, sqrt, asin, floor, pi
import quantum.qubit_abs as qa
import quantum.gate_matrix as gm
from stl.quantum_algs.grovers_alg import (
    grovers_alg,
    optimal_iterations,
    grover_oracle,
    grover_diffuser,
)

np.random.seed(1234)


def exact_success_prob(num_data, num_targets):
    k = optimal_iterations(2 ** num_data, num_targets)
    theta = asin(sqrt(num_targets / 2 ** num_data))
    return k, sin((2 * k + 1) * theta) ** 2


print("=== single marked state, various sizes ===")
for num_data in (2, 3, 4, 5, 6):
    n = num_data + 1
    target = 2 ** num_data - 1  # all ones
    k, expected = exact_success_prob(num_data, 1)

    def circuit(qc, target=target, num_data=num_data):
        def oracle(qc, data_qubits, ancilla):
            qc.apply_gate(gm.px_MAT, [ancilla], control_qubits=data_qubits)
        grovers_alg(qc, oracle)

    qc = qa.QuantumComputer(n)
    counts = qc.sample(circuit, 4000)
    found = sum(c for bits, c in counts.items() if bits[:num_data] == f"{target:0{num_data}b}")
    print(f"  n_data={num_data} k={k} expected~{expected:.3f}  measured={found/4000:.3f}")

print("\n=== two marked states (000 and 111), N=8 ===")
k, expected = exact_success_prob(3, 2)
print(f"  k={k} exact success over the two targets = {expected:.3f}")


def circuit2(qc):
    def oracle(qc, data_qubits, ancilla):
        # mark |000>: ancilla is |->, so X's on controls invert the condition
        for q in data_qubits:
            qc.apply_gate(gm.px_MAT, [q])
        qc.apply_gate(gm.px_MAT, [ancilla], control_qubits=data_qubits)
        for q in data_qubits:
            qc.apply_gate(gm.px_MAT, [q])
        # mark |111>
        qc.apply_gate(gm.px_MAT, [ancilla], control_qubits=data_qubits)
    grovers_alg(qc, oracle, num_targets=2)


qc = qa.QuantumComputer(4)
counts = qc.sample(circuit2, 4000)
hit = sum(c for bits, c in counts.items() if bits[:3] in ("000", "111"))
print(f"  measured success = {hit/4000:.3f}")

print("\n=== oracle with a scratch qubit (compute / kickback / uncompute) ===")
# qubits: q0,q1 data | q2 scratch | q3 ancilla ; mark |11>
num_data = 2
n = num_data + 2
k, expected = exact_success_prob(num_data, 1)


def circuit3(qc):
    scratch = qc[num_data]
    ancilla = qc[num_data + 1]

    def oracle(qc, data_qubits, ancilla):
        l = len(data_qubits)
        # compute scratch = AND(data) with a multi-controlled X
        qc.apply_gate(gm.px_MAT, [scratch], control_qubits=data_qubits)
        # phase kickback
        qc.apply_gate(gm.px_MAT, [ancilla], control_qubits=[scratch])
        # uncompute scratch only
        qc.apply_gate(gm.px_MAT, [scratch], control_qubits=data_qubits)

    grovers_alg(qc, oracle, scratch_qubits=[scratch])


qc = qa.QuantumComputer(n)
counts = qc.sample(circuit3, 4000)
found = sum(c for bits, c in counts.items() if bits[:num_data] == "11")
scratch_dirty = sum(c for bits, c in counts.items() if bits[num_data] == "1")
print(f"  k={k} expected~{expected:.3f}  measured={found/4000:.3f}  scratch-left-dirty={scratch_dirty}")

print("\n=== optimal_iterations sanity ===")
for num_data in (2, 3, 4, 5, 6, 8):
    N = 2 ** num_data
    for M in (1, 2, 3):
        if M > N:
            continue
        k = optimal_iterations(N, M)
        theta = asin(sqrt(M / N))
        p = sin((2 * k + 1) * theta) ** 2
        # brute force global best on a generous range
        best = max(sin((2 * j + 1) * theta) ** 2 for j in range(0, int(pi / (2 * theta)) + 2))
        flag = "" if abs(p - best) < 1e-9 else "  <-- NOT OPTIMAL"
        print(f"  N={N:3d} M={M}: k={k} p={p:.4f}{flag}")