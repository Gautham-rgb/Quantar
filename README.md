# Quantar
A quantum programming language that is a superset of python

## running it

```bash
pip install -e .            # or: uv pip install -e .
python stl/quantum_algs/shors_alg.py
python stl/quantum_algs/grovers_alg.py
```

works from any directory, no `python -m` and no `PYTHONPATH` needed.

optional future backends (pennylane, qiskit, qbraid): `pip install -e ".[backends]"`

## grammar & parser

Quantar is parsed as a strict superset of Python: the parser is generated from
CPython's own `Grammar/python.gram` plus the small operator overlay in
`lang_systems/quantar.gram`. The overlay adds symbolic operators (`a <-> b`) and
identifier operators (`a CNOT b`) registered in `quantar.operators`.

```bash
# regenerate the parser manually (the build hook also does this)
python -m lang_systems.build_parser
```

Use it:

```python
from lang_systems import parse

tree = parse("a <-> b\nx = 1 + 2\n")
```

The generated parser is `lang_systems/quantar_parser.py`; see
`lang_systems/build_parser.py` for the pegen-0.3.0 workarounds. The result is
pegen's raw parse tree; turning it into an AST or bytecode is a separate step.