"""Quantar language systems - parsers and language tools.

``lang_systems.parse`` is the entry point for parsing Quantar source (a strict
superset of Python). It is imported lazily so that importing this package does
not require the generated parser (which is produced at build time) to already
exist.
"""


def __getattr__(name):
    if name == "parse":
        from lang_systems.quantar import parse

        return parse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["parse"]
