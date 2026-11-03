"""Safe recipe loading and deterministic, host-only trial planning."""

import copy
import itertools
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import yaml

from benchmarks.config import resolve_config
from benchmarks.registry import get_definition


class _RecipeLoader(yaml.SafeLoader):
    def get_single_data(self):
        node = self.get_single_node()
        if node is None:
            return None
        _check_aliases(node, set())
        return self.construct_document(node)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            return super().construct_mapping(node, deep=deep)
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Recipe mapping keys must be strings")
            if key in mapping:
                raise ValueError(f"Duplicate recipe key: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _check_aliases(node: yaml.Node, ancestors: set[int]) -> None:
    """Check before SafeLoader flattens merge keys and can erase alias cycles."""
    if isinstance(node, yaml.ScalarNode):
        return
    identity = id(node)
    if identity in ancestors:
        raise ValueError("Recursive YAML aliases are not allowed")
    ancestors.add(identity)
    try:
        children = (
            itertools.chain.from_iterable(node.value)
            if isinstance(node, yaml.MappingNode)
            else node.value
        )
        for child in children:
            _check_aliases(child, ancestors)
    finally:
        ancestors.remove(identity)


def _check_json_data(value: object, ancestors: set[int] | None = None) -> None:
    """Reject YAML values that cannot be represented faithfully in plan.json."""
    if ancestors is None:
        ancestors = set()
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Recursive YAML aliases are not allowed")
        ancestors.add(identity)
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("Recipe mapping keys must be strings")
                children = value.values()
            else:
                children = value
            for child in children:
                _check_json_data(child, ancestors)
        finally:
            ancestors.remove(identity)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Recipe numbers must be finite")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError(f"Unsupported recipe value: {type(value).__name__}")


def _validate_recipe(recipe: object) -> None:
    _check_json_data(recipe)
    schema_path = Path(__file__).resolve().parents[1] / "recipes" / "recipe.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema).validate(recipe)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "recipe"
        raise ValueError(f"Invalid {location}: {error.message}") from error


def _absolute_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _server_args(parameters: dict) -> list[str]:
    arguments = []
    for name, value in sorted(parameters.items()):
        if value is False:
            continue
        arguments.append(name)
        if value is True:
            continue
        if isinstance(value, dict):
            arguments.extend(value["args"])
        else:
            arguments.append(str(value))
    return arguments


def _expand_group(group: dict):
    axes = dict(group["parameters"])
    if "kv_quantization" in group:
        axes["kv_quantization"] = group["kv_quantization"]
    names = sorted(axes)
    choices = [value if isinstance(value, list) else [value] for value in (axes[n] for n in names)]
    for values in itertools.product(*choices):
        parameters = copy.deepcopy(dict(zip(names, values)))
        if "kv_quantization" in parameters:
            precision = parameters.pop("kv_quantization")
            parameters["--cache-type-k"] = precision if isinstance(precision, str) else precision["k"]
            parameters["--cache-type-v"] = precision if isinstance(precision, str) else precision["v"]
        yield dict(sorted(parameters.items()))


def build_plan(
    *,
    recipe_path: Path | None = None,
    models: list[str] | None = None,
    cwd: Path | None = None,
) -> dict:
    """Resolve one input mode without importing runtimes or starting inference.

    Benchmark preparation freezes path-independent workload identity once. Only
    selected benchmarks read inputs; their payloads are not retained in the plan.
    Native defaults are deliberately not materialized as server arguments.
    """
    if (recipe_path is None) == (models is None):
        raise ValueError("Specify exactly one of recipe_path or models")
    working_directory = Path.cwd() if cwd is None else Path(cwd).resolve()
    if recipe_path is not None:
        source_path = _absolute_path(recipe_path, working_directory)
        try:
            recipe = yaml.load(source_path.read_text(encoding="utf-8"), Loader=_RecipeLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid recipe YAML: {error}") from error
        base = source_path.parent
        source = str(source_path)
        default_name = source_path.stem
    else:
        recipe = {
            "schema_version": 1,
            "models": models,
            "inference": {"configurations": [{"parameters": {"--ctx-size": 8192}}]},
        }
        base = working_directory
        source = "cli"
        default_name = "LocalLLMBench"
    _validate_recipe(recipe)

    execution = recipe.get("execution", {})
    shared = resolve_config({
        key: value
        for key, value in recipe.items()
        if key in ("benchmarks", "benchmark_options", "generation", "output_dir")
    } | {"execution": {key: value for key, value in execution.items() if key != "startup_timeout"}})
    shared["execution"]["startup_timeout"] = execution.get("startup_timeout", 300)

    resolved_models = []
    for model in recipe["models"]:
        path = _absolute_path(model, base)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        resolved_models.append(str(path))

    manifests = {}
    for name, options in shared["benchmark_options"].items():
        definition = get_definition(name)
        for field in definition.PATH_OPTIONS:
            options[field] = str(_absolute_path(options[field], base))
    for name in shared["benchmarks"]:
        provenance = get_definition(name).prepare(shared["benchmark_options"][name])[1]
        if not isinstance(provenance, dict):
            raise ValueError(f"{name} preparation provenance must be an object")
        _check_json_data(provenance)
        manifests[name] = copy.deepcopy(provenance)

    trials = []
    configurations = [
        (group.get("name", f"group-{index}"), parameters)
        for index, group in enumerate(recipe["inference"]["configurations"], 1)
        for parameters in _expand_group(group)
    ]
    for model in resolved_models:
        for group, parameters in configurations:
            trials.append({
                "id": f"{len(trials) + 1:04d}",
                "model": model,
                "group": group,
                "parameters": copy.deepcopy(parameters),
                "server_args": _server_args(parameters),
            })
    return {
        "schema_version": 1,
        "name": recipe.get("name", default_name),
        "source": source,
        "output_dir": str(_absolute_path(shared["output_dir"], base)),
        "inference": {
            "bind_address": recipe["inference"].get("bind_address", "127.0.0.1"),
            "port": recipe["inference"].get("port", 8080),
        },
        "benchmarks": shared["benchmarks"],
        "benchmark_options": shared["benchmark_options"],
        "generation": shared["generation"],
        "execution": shared["execution"],
        "manifests": manifests,
        "trials": trials,
    }
