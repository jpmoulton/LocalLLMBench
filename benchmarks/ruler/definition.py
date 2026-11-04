"""Standard-library loading and deterministic selection of frozen RULER prompts."""

import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
)
PROTOCOL = "localllmbench.ruler.retrieval.v1"
GENERATOR_REPOSITORY = "https://github.com/NVIDIA/RULER"
GENERATOR_REVISION = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
SELECTION_ALGORITHM = "localllmbench.ruler.sample.v1"
DEFAULT_OPTIONS = {
    "dataset_path": "C:/Users/hocke/Documents/LLM-Reusable-Assets/benchmarks/datasets/current/ruler/prepared",
    "tasks": list(TASKS),
    "selection": {"count_per_task": 8, "seed": 42},
}
PATH_OPTIONS = ("dataset_path",)
NEEDS_DOCKER = False
SCORE_METRICS = {
    "retrieval_score": {
        "label": "RULER retrieval subset", "direction": "higher", "unit": "percent",
    },
    **{
        f"{task}_score": {
            "label": f"RULER retrieval: {task}", "direction": "higher", "unit": "percent",
        }
        for task in TASKS
    },
}


def _integer(value: object, name: str, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        constraint = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{constraint}")


def _tasks(value: object, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of supported retrieval tasks")
    if any(not isinstance(task, str) or task not in TASKS for task in value):
        raise ValueError(f"{name} must contain only supported retrieval tasks: {', '.join(TASKS)}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate tasks")


def validate_options(options: dict) -> None:
    """Validate configuration without opening datasets or importing runtime packages."""
    if not isinstance(options, dict):
        raise ValueError("ruler options must be a JSON object")
    unknown = options.keys() - DEFAULT_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown ruler options: {', '.join(sorted(unknown))}")
    path = options.get("dataset_path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("ruler.dataset_path must be a nonempty string")
    _tasks(options.get("tasks"), "ruler.tasks")
    selection = options.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("ruler.selection must be a JSON object")
    unknown = selection.keys() - {"count_per_task", "seed"}
    if unknown:
        raise ValueError(f"Unknown ruler.selection options: {', '.join(sorted(unknown))}")
    _integer(selection.get("count_per_task"), "ruler.selection.count_per_task", 1)
    _integer(selection.get("seed"), "ruler.selection.seed")


def _validate_manifest(manifest: Any, corpus_sha256: str) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported RULER manifest schema_version")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"RULER manifest protocol must be {PROTOCOL}")
    if manifest.get("corpus_sha256") != corpus_sha256:
        raise ValueError("manifest corpus_sha256 does not match corpus.jsonl")
    generator = manifest.get("generator")
    if not isinstance(generator, dict) or (
        generator.get("repository") != GENERATOR_REPOSITORY
        or generator.get("revision") != GENERATOR_REVISION
        or not isinstance(generator.get("files_sha256"), dict)
        or not generator["files_sha256"]
    ):
        raise ValueError("manifest generator must identify the pinned NVIDIA RULER source and file hashes")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or (
        tokenizer.get("encoding") != "cl100k_base"
        or tokenizer.get("package") != "tiktoken"
        or not isinstance(tokenizer.get("version"), str)
        or not tokenizer["version"].strip()
    ):
        raise ValueError("manifest tokenizer must identify cl100k_base and an exact tiktoken version")
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("manifest generation must be a JSON object")
    _integer(generation.get("seed"), "manifest generation.seed")
    for key in ("input_tokens", "reserved_output_tokens", "samples_per_task"):
        _integer(generation.get(key), f"manifest generation.{key}", 1)
    _tasks(generation.get("tasks"), "manifest generation.tasks")
    if not isinstance(manifest.get("essays"), (dict, list)) or not manifest["essays"]:
        raise ValueError("manifest essays must record source identity and concatenation policy")
    if not isinstance(manifest.get("dependencies"), dict) or not manifest["dependencies"]:
        raise ValueError("manifest dependencies must record exact generator dependency versions")
    # Also reject non-finite JSON extensions before putting metadata in provenance.
    json.dumps(manifest, allow_nan=False)


def prepare(options: dict) -> tuple[list[dict], dict]:
    """Read a frozen corpus; select a stable per-task hash-ranked prefix.

    Rank is SHA256(UTF8(algorithm + '\\n' + selection_seed + '\\n' + id)),
    with id breaking ties. Neither local paths nor model/generation settings
    participate. Changing selection count preserves each task's ranked prefix.
    Recorded input counts refer to cl100k_base, never the tested model tokenizer.
    """
    validate_options(options)
    directory = Path(options["dataset_path"])
    raw = (directory / "corpus.jsonl").read_bytes()
    corpus_sha256 = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(manifest, corpus_sha256)
    generation = manifest["generation"]
    records_by_task = {task: [] for task in TASKS}
    seen_ids = set()
    seen_indices = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid corpus JSON at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Corpus line {line_number} must be a JSON object")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Corpus line {line_number}: id must be a nonempty string")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        task = record.get("task")
        if not isinstance(task, str) or task not in generation["tasks"]:
            raise ValueError(f"{sample_id}: task must be listed in manifest generation.tasks")
        _integer(record.get("index"), f"{sample_id}: index", 0)
        identity = (task, record["index"])
        if identity in seen_indices:
            raise ValueError(f"Duplicate task/index: {identity}")
        seen_indices.add(identity)
        if not isinstance(record.get("input"), str) or not record["input"].strip():
            raise ValueError(f"{sample_id}: input must be a nonempty string")
        if not isinstance(record.get("answer_prefix"), str):
            raise ValueError(f"{sample_id}: answer_prefix must be a string")
        outputs = record.get("outputs")
        if not isinstance(outputs, list) or not outputs or any(
            not isinstance(answer, str) or not answer.strip() for answer in outputs
        ):
            raise ValueError(f"{sample_id}: outputs must be a nonempty list of nonempty reference strings")
        _integer(record.get("input_tokens"), f"{sample_id}: input_tokens", 1)
        if record["input_tokens"] > generation["input_tokens"]:
            raise ValueError(f"{sample_id}: input_tokens exceeds the reference input budget")
        records_by_task[task].append(record)

    count = options["selection"]["count_per_task"]
    seed = options["selection"]["seed"]

    def rank(record: dict) -> tuple[bytes, str]:
        sample_id = record["id"]
        value = f"{SELECTION_ALGORITHM}\n{seed}\n{sample_id}".encode("utf-8")
        return hashlib.sha256(value).digest(), sample_id

    selected = []
    for task in options["tasks"]:
        population = records_by_task[task]
        if len(population) < count:
            raise ValueError(f"{task}: need {count} samples, found {len(population)} in frozen corpus")
        selected.extend(sorted(population, key=rank)[:count])
    provenance = {
        "protocol": PROTOCOL,
        "algorithm": SELECTION_ALGORITHM,
        "selected_ids": [record["id"] for record in selected],
        "sample_tasks": {record["id"]: record["task"] for record in selected},
        "tasks": list(options["tasks"]),
        "count_per_task": count,
        "selection_seed": seed,
        "input_tokens": generation["input_tokens"],
        "reserved_output_tokens": generation["reserved_output_tokens"],
        "generation_seed": generation["seed"],
        "corpus_sha256": corpus_sha256,
        "source_manifest": manifest,
    }
    return selected, provenance
