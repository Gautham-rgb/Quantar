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

## grammar & parser (known limitation)

The language grammar lives in `lang_systems/quantar.gram` (PEG format).  
A parser is generated at build time via `pegen` and placed at `lang_systems/quantar_parser.py`.

```bash
# regenerate parser manually
pip install pegen
python -m pegen lang_systems/quantar.gram -o lang_systems/quantar_parser.py
```

Use the generated parser:

```python
from lang_systems.quantar_parser import GeneratedParser
from pegen.tokenizer import Tokenizer
import tokenize, io

code = "qc.apply(q0)\nqc.H(q0)\n"
tokengen = tokenize.generate_tokens(io.StringIO(code).readline)
parser = GeneratedParser(Tokenizer(tokengen))
result = parser.start()
```

**Known limitation**: The PEG grammar uses `expr: atom ('.' atom)*` with `atom: '(' expr ')'` which creates a rule cycle that triggers pegen's overzealous left-recursion detection. The parser works for simple expressions but fails on attribute access (`qc.apply(q0)`) and method calls. This is a known pegen limitation with rule cycles. The parser works correctly for simple expressions without parentheses.

The build hook (`hatch_build.py`) regenerates the parser automatically when building the package.