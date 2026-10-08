# Quantar
A quantum programming language that is a superset of python

## running it

```bash
pip install -e .            # or: uv pip install -e .
python stl/quantum_algs/shors_alg.py
python stl/quantum_algs/grovers_alg.py
python tests/test_qft_fuse.py
```

works from any directory, no `python -m` and no `PYTHONPATH` needed.

optional future backends (pennylane, qiskit, qbraid): `pip install -e ".[backends]"`
