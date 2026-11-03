"""Host-safe BigCodeBench corpus loading and reproducible selection."""

import hashlib
import json
import math
from pathlib import Path


SELECTION_ALGORITHM = "localllmbench.bigcodebench.sample.v1"
DEFAULT_OPTIONS = {
    "dataset_path": "C:/Users/hocke/Documents/LLM-Reusable-Assets/benchmarks/datasets/current/bigcodebench",
    "selection": {"count": 64, "seed": 42},
    "test_timeout": 60,
}
PATH_OPTIONS = ("dataset_path",)
NEEDS_DOCKER = True


def _integer(value: object, name: str, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        constraint = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{constraint}")


def validate_options(options: dict) -> None:
    """Validate merged options without accessing the dataset."""
    if not isinstance(options, dict):
        raise ValueError("bigcodebench options must be a JSON object")
    unknown = options.keys() - DEFAULT_OPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown bigcodebench options: {', '.join(sorted(unknown))}")
    dataset_path = options.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError("bigcodebench.dataset_path must be a nonempty string")
    selection = options.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("bigcodebench.selection must be a JSON object")
    unknown = selection.keys() - {"count", "seed"}
    if unknown:
        raise ValueError(f"Unknown bigcodebench.selection options: {', '.join(sorted(unknown))}")
    _integer(selection.get("count"), "bigcodebench.selection.count", 1)
    _integer(selection.get("seed"), "bigcodebench.selection.seed")
    timeout = options.get("test_timeout")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("bigcodebench.test_timeout must be a finite number > 0")


def prepare(options: dict) -> tuple[list[dict], dict]:
    """Load the selected corpus and its benchmark-owned provenance."""
    validate_options(options)
    return select_dataset(Path(options["dataset_path"]), **options["selection"])


def select_dataset(directory: Path, count: int, seed: int) -> tuple[list[dict], dict]:
    """Select a nested hash-ranked prefix from corpus.jsonl, never sample.jsonl.

    Rank is SHA256(UTF8(algorithm + '\\n' + seed + '\\n' + task_id)), with
    task_id breaking digest ties. The canonical fingerprint hashes the compact,
    sorted-key, UTF-8 JSON array of all records sorted by task_id (no newline).
    Manifest metadata describes the corpus, not any pre-existing sample policy.
    """
    _integer(count, "count", 1)
    _integer(seed, "seed")
    raw_corpus = (directory / "corpus.jsonl").read_bytes()
    corpus_sha256 = hashlib.sha256(raw_corpus).hexdigest()
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if "corpus_sha256" in manifest and manifest["corpus_sha256"] != corpus_sha256:
        raise ValueError("manifest corpus_sha256 does not match corpus.jsonl")
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("manifest source must be a JSON object")

    records = []
    seen_ids = set()
    for line_number, line in enumerate(raw_corpus.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid corpus JSON at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Corpus line {line_number} must be a JSON object")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Corpus line {line_number} needs a nonempty task_id")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        seen_ids.add(task_id)
        for field in ("complete_prompt", "canonical_solution", "test"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{task_id}: {field} must be a nonempty string")
        libs = record.get("libs")
        if not (
            (isinstance(libs, str) and libs.strip())
            or (
                isinstance(libs, list)
                and all(isinstance(lib, str) and lib.strip() for lib in libs)
            )
        ):
            raise ValueError(f"{task_id}: libs must be a nonempty string or list of strings")
        records.append(record)

    if count > len(records):
        raise ValueError(f"count must be in 1..{len(records)}; received {count}")
    canonical_corpus = json.dumps(
        sorted(records, key=lambda record: record["task_id"]),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    def rank(record: dict) -> tuple[bytes, str]:
        task_id = record["task_id"]
        payload = f"{SELECTION_ALGORITHM}\n{seed}\n{task_id}".encode("utf-8")
        return hashlib.sha256(payload).digest(), task_id

    selected = sorted(records, key=rank)[:count]
    provenance = {
        "algorithm": SELECTION_ALGORITHM,
        "count": count,
        "seed": seed,
        "selected_ids": [record["task_id"] for record in selected],
        "population": len(records),
        "corpus_sha256": corpus_sha256,
        "corpus_fingerprint_sha256": hashlib.sha256(canonical_corpus).hexdigest(),
        "dataset_version": manifest.get("dataset_version", source.get("dataset_split")),
        "dataset_revision": source.get(
            "dataset_revision", manifest.get("dataset_revision", manifest.get("revision"))
        ),
    }
    return selected, provenance
