# Devlog 3: gate programs, fusion, and a qft that was wrong

devlog 2 made gates fast. this one found out the qft/iqft circuits and shor's quantum path were quietly wrong, fixed both, and taught the simulator to apply whole runs of gates in a single pass.

## 1) programs as gates

`apply_gate` stays public, `op_hist` stays append-only, and a program is now just a list of `Gate`s:

- `qc.gate(...)` builds a gate record and returns it (this is what `apply_gate` uses internally)
- `qc.apply(...)` accepts a program list, a single `Gate`, or a callable composite; parent `control_qubits` compose onto every gate in a program (targets-with-a-program still raises)
- `CircuitGate.build(name, width, builder)` runs `builder` on a scratch machine, freezes its `op_hist` as one reusable gate; `.inverse()` reverses + inverts it
- `CircuitGate(qc, targets, controls)` remaps indices (parent controls appended) and replays flat through `qc._apply_program` - controlled replay records flat, exports flat, no nested history

`qft_program`/`iqft_program` return gate lists, `qft_circuit`/`iqft_circuit` are one `qc.apply` call, and `qft_gate(n)`/`iqft_gate(n)` are cached `CircuitGate`s. `_validate_gate` now runs for every gate *before* any state change, so a dead qubit in gate 40 of a program raises with state and history untouched.

## 2) fusion: one pass per run, not per gate

after validation, `_apply_program` scans consecutive runs:

- **diagonal runs** (the cp ladder of a qft) -> one pass, per-basis product of the phase factors
- **swap runs** (the bit-reversal layer) -> one permutation pass, each amplitude moves exactly once
- anything else -> `_run` as before

`op_hist` receives the individual gates either way, so `uncompute` and qasm export still see `cp` after `cp`. overlapping swaps stop the run (they don't commute). numpy fallbacks exist for both fused paths when numba is missing.

numba quirk worth recording: under `prange`, a loop index carried through a loop-reassigned alias gets inferred as float64 ("no implementation of rshift(float64, int64)"). `swapped = np.int64(basis)` fixes it.

## 3) the qft was computing bit-reversed input

devlog 2 claimed "QFT/IQFT still match the dense matrices exactly". that claim was wrong. measured against `gm.qft_MAT`, the old circuit computed `qft_MAT @ bitrev(input)` - the input was read in reversed order. it never crashed anything because the pair still round-tripped (iqft undid the same error) and shor's subroutine never got as far as using the output (next section).

we tried to fix it without swaps: four loop orders (H ascending/descending x cp before/after) all fail vs the dense matrix, errors 0.3-0.7. there is no swap-free qft here. what works: bit-reversal swaps *first*, then the original sequence -> exact `qft_MAT`; `iqft` is the exact reverse-inverse (negated phases, H, swaps last). worst error over n=1..6 is now 2.4e-14, round trip 3.9e-16.

the swaps cost n/2 extra passes, which is why they got fused (section 2).

## 4) shor's quantum path was dead

`quantum_subroutine` argmax'd the probabilities - but the iqft output has equal peaks at y=0 and at every m*2^t/r, and argmax ties resolve to the lowest index. it always returned y=0, `pow(a, 0, N)` is 1, `gcd(0, N)` is N, retry. before this fix `shors_alg` only ever factored via the classical gcd shortcut; the quantum subroutine contributed nothing. fix: mask everything `>= 2^m` (the k=0 row) to -1 before argmax. peaks now hit for a=7, 2, 11 on N=15, and 91 factors through the quantum path in 0.4 s (seed-dependent; some draws still hit the gcd shortcut first).

## 5) qasm export fixes

- `cp 0.785, q[0], q[1]` was invalid openqasm3; now `cp(0.785) q[0], q[1]`
- `U(theta, phi, lambda)` exported the literal words instead of numbers; one branch now emits real params
- the `name_is_controlled` skip-rule keeps `cx`/`ccx` free of `ctrl @` while `cp` with extra controls gets it; the export parses with the `openqasm3` package

## 6) results

| iqft run | gates | one-at-a-time | fused |
|---|---|---|---|
| t=8  (n=8)  | 40  | 0.001 s | 0.001 s (1.6x) |
| t=14 (n=21) | 112 | 0.754 s | 0.216 s (3.5x) |
| t=16 (n=24) | 144 | 8.168 s | 1.989 s (4.1x) |

fused output identical to gate-for-gate, max diff ~1e-18. t=14 IQFT across all three devlogs: 7.95 s -> 3.05 s -> 0.216 s. `shors_alg(15/21/35)` still factors, the 60-test suite (dense qft/iqft, fusion equivalence, CircuitGate replay/controls, validation atomicity, uncompute, qasm parse, shor peaks) is green, and grover + kernel-vs-dense regressions pass.

## 7) still open

- `state_vector` is machine-owned working memory: the fused/gated paths ping-pong it with `_scratch`, so a buffer shared across two machines gets clobbered on the second gate. tests must give each machine its own array. the library behavior is correct; the aliasing is on the caller
- 3+ targets still go through the dense path; modular exponentiation in shor is still classical (unchanged since devlog 1)

## 8) packaging follow-up: run it normally

the project was never installable, which is why every run needed `python -m ...` or a `PYTHONPATH`: `dynamic = ["__version__"]` (hatch wants `version`), the version path pointed at a `Quantar/` folder that does not exist, and nothing told hatch which packages to build. all three fixed: version reads the root `__init__.py`, the wheel ships `quantum/` + `stl/` (both now regular packages with `__init__.py`), `openqasm3` is declared as `openqasm3[parser]` (1.0 moved the parser into an extra - without it the "openqasm3 parses the export" check fails), and the heavy future backends (pennylane/qiskit/qbraid/qiskit-aer) moved to a `[backends]` extra so a plain `pip install -e .` only pulls numpy + numba + openqasm3.

the check scripts moved into `tests/` and lost their `sys.path.insert(0, ...)` hacks. after one editable install you run `python tests/test_qft_fuse.py` or `python stl/quantum_algs/shors_alg.py` from anywhere - no `-m`, no PYTHONPATH. `tests/conftest.py` keeps pytest from importing those scripts (they are scripts: run them, don't collect them). this closes the version-path issue open since devlog 1.
