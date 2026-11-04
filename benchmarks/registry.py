"""Explicit benchmark names and lazy host/runtime module imports."""

from importlib import import_module
from types import ModuleType


BENCHMARKS = {
    "bigcodebench": "benchmarks.bigcodebench",
    "ruler": "benchmarks.ruler",
}


def resolve_benchmarks(names: list[str] | None) -> list[str]:
    """Return all registered names by default, or a validated ordered subset."""
    if names is None:
        return list(BENCHMARKS)
    if not isinstance(names, list) or not names:
        raise ValueError("benchmarks must be a nonempty list of benchmark names")
    seen = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("benchmarks must contain nonempty string names")
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {name}")
        if name in seen:
            raise ValueError(f"Duplicate benchmark: {name}")
        seen.add(name)
    return list(names)


def get_definition(name: str) -> ModuleType:
    """Import a benchmark's standard-library-only host definition."""
    resolve_benchmarks([name])
    return import_module(f"{BENCHMARKS[name]}.definition")


def get_runtime(name: str) -> ModuleType:
    """Import a benchmark's Inspect runtime only when the controller needs it."""
    resolve_benchmarks([name])
    return import_module(f"{BENCHMARKS[name]}.task")
