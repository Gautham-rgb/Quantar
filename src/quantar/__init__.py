"""Quantar umbrella package.

The implementation lives in the ``lang_systems``, ``quantum``, and ``stl``
packages. Importing ``quantar`` stays lightweight; subpackages and common
objects are loaded on first use.
"""

from importlib import import_module

from lang_systems import __version__

_SUBPACKAGES = ("lang_systems", "quantum", "stl")

_EXPORTS = {
    "parse": ("lang_systems", "parse"),
    "parse_ast": ("lang_systems", "parse_ast"),
    "build_ast": ("lang_systems", "build_ast"),
    "lower_ast": ("lang_systems", "lower_ast"),
    "compile_ast": ("lang_systems", "compile_ast"),
    "QuantumOp": ("lang_systems", "QuantumOp"),
    "load_operators": ("lang_systems", "load_operators"),
    "list_known_operators": ("lang_systems", "list_known_operators"),
    "QuantumComputer": ("quantum", "QuantumComputer"),
    "Qubit": ("quantum", "Qubit"),
    "Gate": ("quantum", "Gate"),
    "CompositeGate": ("quantum", "CompositeGate"),
    "CircuitGate": ("quantum", "CircuitGate"),
    "named_gate": ("quantum", "named_gate"),
    "to_openqasm3": ("quantum", "to_openqasm3"),
    "crk": ("quantum", "crk"),
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        value = import_module(name)
    else:
        target = _EXPORTS.get(name)
        if target is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module_name, attribute = target
        value = getattr(import_module(module_name), attribute)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["__version__", *_SUBPACKAGES, *_EXPORTS]
