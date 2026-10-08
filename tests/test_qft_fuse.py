"""Verify CircuitGate, fused diagonal runs, and the fused QFT/IQFT programs."""

import time

import numpy as np

import quantum.gate_matrix as gm
import quantum.qubit_abs as qa
from stl.quantum_algs.shors_alg import (
    qft_circuit,
    iqft_circuit,
    qft_program,
    iqft_program,
    qft_gate,
    iqft_gate,
    shors_alg,
)

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f" :: {detail}"))
    if not ok:
        failures.append(name)


rng = np.random.default_rng(7)


def random_state(n):
    state = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return (state / np.linalg.norm(state)).astype(complex)


def loaded(n, state=None):
    # Each machine owns its buffer: the simulator ping-pongs state through a
    # scratch array, so sharing one `state` across machines corrupts it.
    qc = qa.QuantumComputer(n)
    source = state if state is not None else random_state(n)
    qc.state_vector = np.array(source, dtype=complex).reshape(-1, 1)
    return qc


# --- 1. QFT / IQFT against the dense matrices -------------------------------
for n in range(1, 7):
    state = random_state(n)

    forward = loaded(n, state)
    qft_circuit(forward, forward.qubits)
    err = np.max(np.abs(forward.state_vector.reshape(-1) - gm.qft_MAT(n) @ state))
    check(f"qft n={n} == dense", err < 1e-12, f"max err {err:.3e}")

    backward = loaded(n, state)
    iqft_circuit(backward, backward.qubits)
    err = np.max(np.abs(backward.state_vector.reshape(-1) - gm.iqft_MAT(n) @ state))
    check(f"iqft n={n} == dense", err < 1e-12, f"max err {err:.3e}")

    round_trip = loaded(n, state)
    qft_circuit(round_trip, round_trip.qubits)
    iqft_circuit(round_trip, round_trip.qubits)
    err = np.max(np.abs(round_trip.state_vector - state.reshape(-1, 1)))
    check(f"qft->iqft round trip n={n}", err < 1e-12, f"max err {err:.3e}")

# --- 2. fused execution == one gate at a time -------------------------------
for n in (3, 4, 5):
    state = random_state(n)

    fused = loaded(n, state)
    program = qft_program(fused, fused.qubits)
    fused._apply_program(program)

    slow = loaded(n, state)
    for gate in program:
        slow._run(gate)

    err = np.max(np.abs(fused.state_vector - slow.state_vector))
    check(f"qft fused == one-at-a-time n={n}", err < 1e-13, f"max err {err:.3e}")
    check(f"program recorded gate-for-gate n={n}", len(fused.op_hist) == len(program))

    fused_i = loaded(n, state)
    program_i = iqft_program(fused_i, fused_i.qubits)
    fused_i._apply_program(program_i)

    slow_i = loaded(n, state)
    for gate in program_i:
        slow_i._run(gate)

    err = np.max(np.abs(fused_i.state_vector - slow_i.state_vector))
    check(f"iqft fused == one-at-a-time n={n}", err < 1e-13, f"max err {err:.3e}")

# --- 3. CircuitGate: a QuantumComputer used as a gate -----------------------
for width in (1, 2, 4):
    state = random_state(width)

    by_hand = loaded(width, state)
    qft_circuit(by_hand, by_hand.qubits)

    as_gate = loaded(width, state)
    as_gate.apply(qft_gate(width), as_gate.qubits)
    check(
        f"qft_gate({width}) replay == circuit",
        np.allclose(as_gate.state_vector, by_hand.state_vector),
    )

    round_trip = loaded(width, state)
    round_trip.apply(qft_gate(width), round_trip.qubits)
    round_trip.apply(iqft_gate(width), round_trip.qubits)
    err = np.max(np.abs(round_trip.state_vector - state.reshape(-1, 1)))
    check(f"qft_gate+iqft_gate round trip width={width}", err < 1e-12, f"max err {err:.3e}")

# controlled replay: every inner gate gets the parent control appended
state = random_state(3)
gated = loaded(3, state)
gadget = qft_gate(2)
gated.apply(gadget, [gated[1], gated[2]], control_qubits=[gated[0]])

manual = loaded(3, state)
for inner in gadget.program:
    manual._run(inner.remap([1, 2], [0]))
check("controlled qft_gate == manual", np.allclose(gated.state_vector, manual.state_vector))
check("controlled replay recorded flat", len(gated.op_hist) == len(gadget.program))

# --- 4. programs with parent controls, and operand errors -------------------
state = random_state(3)
controlled = loaded(3, state)
controlled.apply(qft_program(controlled, controlled.qubits[:2]), control_qubits=[controlled[2]])

manual = loaded(3, state)
for gate in qft_program(manual, manual.qubits[:2]):
    manual._run(gate.remap([0, 1, 2], [2]))
check("program with parent control", np.allclose(controlled.state_vector, manual.state_vector))

try:
    qa.QuantumComputer(2).apply([qa.Gate(gm.h_MAT, [], [0])], [0])
    check("targets with a program raise", False)
except ValueError:
    check("targets with a program raise", True)

try:
    qa.QuantumComputer(2).apply(["nope"])
    check("non-gate program entry raises", False)
except TypeError:
    check("non-gate program entry raises", True)

try:
    qa.QuantumComputer(4).apply(qft_gate(3), [0])
    check("width mismatch raises", False)
except ValueError:
    check("width mismatch raises", True)

# validation runs before anything is applied
state = random_state(2)
guarded = loaded(2, state)
guarded.qubits[1].is_measured = True
before = guarded.state_vector.copy()
program = [guarded.gate(gm.h_MAT, [0]), guarded.gate(gm.h_MAT, [1])]
try:
    guarded._apply_program(program)
    check("dead qubit in program raises", False)
except ValueError:
    check("dead qubit in program raises", True)
check("state untouched by rejected program", np.allclose(guarded.state_vector, before))
check("history untouched by rejected program", guarded.op_hist == [])

try:
    qa.QuantumComputer(2)._apply_program([qa.Gate(gm.h_MAT, [0], [0])])
    check("overlapping control/target raises", False)
except ValueError:
    check("overlapping control/target raises", True)

# --- 5. uncompute and export over a fused program ---------------------------
qc = qa.QuantumComputer(2)
qc.apply_gate(gm.h_MAT, [0])
mark = qc.checkpoint()
qc.apply(qft_program(qc, qc.qubits))
qc.uncompute([0, 1], checkpoint=mark)

reference = qa.QuantumComputer(2)
reference.apply_gate(gm.h_MAT, [0])
check("uncompute undoes a program", np.allclose(qc.state_vector, reference.state_vector))
check(
    "uncompute appends inverses",
    len(qc.op_hist) == 1 + 2 * len(qft_program(qc, qc.qubits)),
)

export = qa.QuantumComputer(3)
qft_circuit(export, export.qubits)
source = export.to_openqasm3()
check("export spells cp(...)", "cp(" in source and "cp 0." not in source)
try:
    import openqasm3

    openqasm3.parse(source)
    check("openqasm3 parses the export", True)
except ImportError:
    print("SKIP openqasm3 not installed")
except Exception as exc:
    check("openqasm3 parses the export", False, str(exc).splitlines()[0])

# --- 6. Shor's quantum path picks a real peak -------------------------------
from stl.quantum_algs.shors_alg import quantum_subroutine

for n_val, a in ((15, 7), (15, 2), (15, 11), (21, 5), (35, 3), (91, 5)):
    m_bits = n_val.bit_length()
    t_bits = 2 * m_bits
    order = 1
    while pow(a, order, n_val) != 1:
        order += 1
    if (1 << t_bits) % order or order == 1:
        print(f"SKIP peaks N={n_val} a={a}: order {order} does not divide 2**{t_bits}")
        continue
    y = int(quantum_subroutine(n_val, a), 2)
    peaks = {j * (1 << t_bits) // order for j in range(1, order)}
    check(f"subroutine N={n_val} a={a} hits a peak", y in peaks, f"y={y}, peaks={sorted(peaks)}")

start = time.perf_counter()
for n_val in (15, 21, 35):
    result = shors_alg(n_val)
    check(
        f"shors_alg({n_val}) factors",
        result is not None and result[0] * result[1] == n_val,
        str(result),
    )
print(f"BENCH shors_alg(15/21/35) in {time.perf_counter() - start:.3f}s")

# --- 7. benchmarks: fused vs one-at-a-time ----------------------------------
def bench(t, total, label):
    state = random_state(total)
    fused = loaded(total, state)
    program = iqft_program(fused, fused.qubits[:t])

    start = time.perf_counter()
    fused._apply_program(program)
    fused_time = time.perf_counter() - start

    slow = loaded(total, state)
    start = time.perf_counter()
    for gate in program:
        slow._run(gate)
    slow_time = time.perf_counter() - start

    err = np.max(np.abs(fused.state_vector - slow.state_vector))
    print(
        f"BENCH {label}: t={t} n={total} gates={len(program)} "
        f"fused={fused_time:.3f}s one-at-a-time={slow_time:.3f}s "
        f"speedup={slow_time / fused_time:.1f}x max diff={err:.2e}"
    )
    check(f"bench output identical t={t}", err < 1e-12)


bench(8, 8, "shor t=8")
bench(14, 21, "shor m=7 t=14 (N=91)")
bench(16, 24, "shor m=8 t=16 (N=143)")

start = time.perf_counter()
result = shors_alg(91)
print(f"BENCH shors_alg(91) = {result} in {time.perf_counter() - start:.3f}s")
check("shors_alg(91) factors", result is not None and result[0] * result[1] == 91)

print()
print("FAILURES:", failures if failures else "none")
