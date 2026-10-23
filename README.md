# Quantar

Quantar is a strict superset of Python for quantum computing and physics
simulations. It keeps Python syntax and adds custom infix operators such as
`a <-> b`, plus a library of quantum gates, circuits, algorithms, and physics
helpers.

## Requirements

- Python 3.12 or newer
- NumPy, Numba, and `openqasm3[parser]` (installed automatically)

## Installation

```bash
pip install quantar

# or with uv
uv pip install quantar
```

Optional extras:

```bash
# alternative quantum backends
pip install quantar[backends]

# SciPy-powered COBYLA optimization
pip install quantar[optimization]

# LSP support
pip install quantar[lsp]
```

## Parsing Quantar

`lang_systems.parse()` returns pegen's raw parse tree. This is the low-level
form used by tooling and tests:

```python
from lang_systems import parse

tree = parse("a <-> b\nx = 1 + 2\n")
```

`lang_systems.parse_ast()` returns a normal Python AST with custom `QuantumOp`
nodes:

```python
import ast

from lang_systems import OperatorCollector, parse_ast

module = parse_ast("x = a <-> b\n")
print(ast.dump(module, indent=2))

collector = OperatorCollector()
collector.visit(module)
print(collector.operators)  # [("<->", "binary")]
```

Custom operators are registered with `lang_systems.operators.operator`:

```python
from lang_systems.operators import operator

@operator.binary("><")
def reverse(left, right):
    return right, left
```

If the operators live in another module, import that module before parsing:

```python
from lang_systems import load_operators, parse

load_operators("my_package.quantar_ops")
tree = parse("a >< b\n")
```

## Quantar as a library

The quantum and physics packages can be imported directly from Python:

```python
import quantum
from stl.quantum_algs.shors_alg import shors_alg
from stl.physics_algs.units import Quantity, Unit
```

See the modules under `quantum/` and `stl/` for the available gates, circuits,
algorithms, and physics helpers.

## Development

Regenerate the parser after changing `lang_systems/quantar.gram`:

```bash
python -m lang_systems.pegen_gram_gen
```

Build the wheel and source distribution:

```bash
uv build
```

The generated parser is `lang_systems/quantar_parser.py`; the build hook in
`hatch_build.py` regenerates it automatically. The parser is generated from
CPython's `Grammar/python.gram` plus the operator overlay in
`lang_systems/quantar.gram`.

## Status

Quantar currently provides the parser, AST, quantum library, and physics
library. A CLI, bytecode interpreter, and editor extension are not implemented
yet.

## Feedback

Feedback and bug reports are welcome through GitHub issues or by contacting the
maintainer. The main contributor is
[Gautham_rgb](https://github.com/Gautham-rgb/).

## Notes on AI

Most of the code was written by people. Some AI assistance was used for
boilerplate and for explaining the mathematics behind topics such as Kronecker
products and gate matrices.
