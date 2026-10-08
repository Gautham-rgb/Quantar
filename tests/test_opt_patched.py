import numpy as np
import quantum.qubit_abs as qa
import quantum.gate_matrix as gm

np.random.seed(7)


def numpy_reference_apply(state_vector, gate_matrix, num_qubits, crtl_pos, targ_pos):
    dim = 1 << num_qubits
    new_state = np.zeros(dim, dtype=complex)
    targ_pos = [int(p) for p in targ_pos]
    cmask = 0
    for p in targ_pos:
        cmask |= (1 << p)
    for i in range(dim):
        amp = state_vector[i, 0]
        if amp == 0:
            continue
        if any(((i >> pos) & 1) == 0 for pos in crtl_pos):
            new_state[i] += amp
            continue
        targ_state = 0
        for pos in targ_pos:
            targ_state = (targ_state << 1) | ((i >> pos) & 1)
        for j in range(1 << len(targ_pos)):
            new_bits = 0
            for k, pos in enumerate(targ_pos):
                new_bits |= ((j >> (len(targ_pos) - 1 - k)) & 1) << pos
            new_state[(i & ~cmask) | new_bits] += gate_matrix[j, targ_state] * amp
    return new_state.reshape(-1, 1)


POOL = {
    2: {"H": gm.h_MAT, "X": gm.px_MAT, "Y": gm.py_MAT, "Z": gm.pz_MAT,
        "RX": gm.rx_MAT(0.7), "RY": gm.ry_MAT(0.9), "RZ": gm.rz_MAT(1.3)},
    4: {"SWAP": gm.swap_MAT, "ISWAP": gm.iswap_MAT, "SQSWAP": gm.swap_sqrt_MAT},
}

print("=== kernel vs numpy reference (disjoint ctrl/target) ===")
mismatch = 0
for n in (1, 2, 3, 4, 5, 6):
    for trial in range(80):
        v = np.random.randn(1 << n) + 1j * np.random.randn(1 << n)
        v /= np.linalg.norm(v)
        state = v.reshape(-1, 1).astype(complex)

        nq = int(np.random.randint(1, min(2, n) + 1))
        qubits = list(np.random.permutation(n))
        tg = qubits[:nq]
        remaining = qubits[nq:]
        nc = int(np.random.randint(0, len(remaining) + 1))
        ct = remaining[:nc]
        gate = POOL[1 << nq][list(POOL[1 << nq].keys())[int(np.random.randint(len(POOL[1 << nq])))]]

        crtl = np.array(sorted((n - 1 - int(q) for q in ct)), dtype=np.int64)
        targ = np.array([n - 1 - int(q) for q in tg], dtype=np.int64)
        ref = numpy_reference_apply(state, gate.astype(complex), n, crtl, targ)

        qc = qa.QuantumComputer(n)
        qc.state_vector = state.copy()
        qc.apply_gate(gate, [qc[int(q)] for q in tg], [qc[int(q)] for q in ct], log_history=False)
        err = float(np.max(np.abs(ref - qc.state_vector)))
        if err > 1e-9:
            mismatch += 1
            if mismatch <= 10:
                print(f"  MISMATCH n={n} targ={tg} ctrl={ct} err={err:.4g}")
print(f"  mismatches: {mismatch}")

print("\n=== unitarity ===")
for n in (2, 3, 4, 5, 6, 8):
    qc = qa.QuantumComputer(n)
    for q in qc.qubits:
        qc.apply_gate(gm.h_MAT, [q], log_history=False)
    for q in qc.qubits:
        qc.apply_gate(gm.rx_MAT(0.4 + 0.1 * q.index), [q], log_history=False)
    for i in range(0, n - 1, 2):
        qc.apply_gate(gm.swap_MAT, [qc[i], qc[i + 1]], log_history=False)
    p = float((np.abs(qc.state_vector.flatten()) ** 2).sum())
    print(f"  n={n}: sum={p:.10f}")

print("\n=== in-place leaves old references valid? (state_vector identity) ===")
qc = qa.QuantumComputer(3)
before = id(qc.state_vector)
qc.apply_gate(gm.h_MAT, [qc[0]])
print(f"  id stable: {before == id(qc.state_vector)}")

print("\n=== uncompute still works ===")
qc = qa.QuantumComputer(3)
for q in qc.qubits:
    qc.apply_gate(gm.h_MAT, [q])
checkpoint = qc.checkpoint()
qc.apply_gate(gm.px_MAT, [qc[2]], [qc[0], qc[1]])  # Toffoli
qc.apply_gate(gm.pz_MAT, [qc[1]])
qc.uncompute([qc[0], qc[1], qc[2]], checkpoint)
expected = np.ones(8, dtype=complex) / np.sqrt(8)
print(f"  restored superposition: {np.max(np.abs(qc.state_vector.flatten() - expected)):.2e}")

print("\n=== fused diffuser == gate diffuser ===")
for n in (3, 4, 5):
    d = n - 1
    v = np.random.randn(1 << n) + 1j * np.random.randn(1 << n)
    v /= np.linalg.norm(v)

    ref = qa.QuantumComputer(n)
    ref.state_vector = v.reshape(-1, 1).astype(complex).copy()
    for q in ref[0:d]:
        ref.apply_gate(gm.h_MAT, [q]); ref.apply_gate(gm.px_MAT, [q])
    ref.apply_gate(gm.pz_MAT, [ref[0]], control_qubits=ref[1:d])
    for q in ref[0:d]:
        ref.apply_gate(gm.px_MAT, [q])
    for q in ref[0:d]:
        ref.apply_gate(gm.h_MAT, [q])

    from stl.quantum_algs.grovers_alg import grover_diffuser
    fused = qa.QuantumComputer(n)
    fused.state_vector = v.reshape(-1, 1).astype(complex).copy()
    grover_diffuser(fused, fused[0:d])

    print(f"  n={n}: max diff = {np.max(np.abs(ref.state_vector - fused.state_vector)):.2e}")
