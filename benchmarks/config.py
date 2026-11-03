"""Shared run configuration with benchmark-owned options and validation."""

import copy
import json
import math
from pathlib import Path

from benchmarks.registry import get_definition, resolve_benchmarks


DEFAULT_CONFIG = {
    "benchmarks": None,
    "benchmark_options": {},
    "output_dir": "results",
    "inference": {
        "container": "localllmbench-inference-128k",
        "model": "local-model",
    },
    "generation": {
        "thinking": True,
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "seed": 42,
    },
    "execution": {"concurrency": 1, "request_timeout": 900},
}


def _merge(defaults: dict, overrides: dict, prefix: str = "") -> dict:
    if not isinstance(overrides, dict):
        raise ValueError(f"{prefix or 'config'} must be a JSON object")
    unknown = overrides.keys() - defaults.keys()
    if unknown:
        names = ", ".join(f"{prefix}{name}" for name in sorted(unknown))
        raise ValueError(f"Unknown configuration keys: {names}")
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(defaults[key], dict):
            merged[key] = _merge(defaults[key], value, f"{prefix}{key}.")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _integer(value: object, name: str, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        constraint = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{constraint}")


def _number(
    value: object,
    name: str,
    minimum: float,
    maximum: float | None = None,
    *,
    exclusive_minimum: bool = False,
) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or (exclusive_minimum and value == minimum):
        operator = ">" if exclusive_minimum else ">="
        raise ValueError(f"{name} must be a finite number {operator} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def resolve_config(overrides: dict) -> dict:
    """Merge and validate shared settings and registered benchmark options.

    Omitted or null benchmark selection resolves to all current registry names.
    Definitions provide defaults on every call; no datasets or runtime modules
    are loaded. Relative paths remain unresolved for the host launcher. Unknown
    keys and invalid values fail, and each returned configuration is independent.
    """
    if not isinstance(overrides, dict):
        raise ValueError("config must be a JSON object")
    selected = resolve_benchmarks(overrides.get("benchmarks"))
    definitions = {name: get_definition(name) for name in resolve_benchmarks(None)}
    defaults = copy.deepcopy(DEFAULT_CONFIG)
    defaults["benchmark_options"] = {
        name: definition.DEFAULT_OPTIONS for name, definition in definitions.items()
    }
    config = _merge(defaults, overrides)
    config["benchmarks"] = selected
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty string")
    inference = config["inference"]
    for key in ("container", "model"):
        if not isinstance(inference[key], str) or not inference[key].strip():
            raise ValueError(f"inference.{key} must be a nonempty string")
    generation = config["generation"]
    if type(generation["thinking"]) is not bool:
        raise ValueError("generation.thinking must be a boolean")
    _integer(generation["max_tokens"], "generation.max_tokens", 1)
    _integer(generation["seed"], "generation.seed")
    _integer(generation["top_k"], "generation.top_k", 0)
    _number(generation["temperature"], "generation.temperature", 0, 2)
    _number(generation["top_p"], "generation.top_p", 0, 1, exclusive_minimum=True)
    _number(generation["min_p"], "generation.min_p", 0, 1)
    execution = config["execution"]
    _integer(execution["concurrency"], "execution.concurrency", 1)
    _number(execution["request_timeout"], "execution.request_timeout", 0, exclusive_minimum=True)
    for name, definition in definitions.items():
        definition.validate_options(config["benchmark_options"][name])
    return config


def load_config(path: Path) -> dict:
    """Read a standalone JSON configuration and resolve its shared defaults."""
    return resolve_config(json.loads(path.read_text(encoding="utf-8")))
