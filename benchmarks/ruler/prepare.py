"""Prepare the local-essay RULER retrieval subset without model weights or GPUs.

Run in the rebuilt benchmark image with its entrypoint overridden to ``python``:
``python -m benchmarks.ruler.prepare --essays /ruler/essays --output /ruler/prepared``.
The output directory must not exist. Only temporary copies of essays are used.

NVIDIA RULER's Apache-2.0 generator is fetched, verified and run unmodified.
Its official configuration and base template are used, but the input tokenizer,
local essay corpus and reserved output budget are explicit local adaptations.
See THIRD_PARTY_NOTICES.txt for upstream attribution and license terms.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from .definition import GENERATOR_REPOSITORY, GENERATOR_REVISION, PROTOCOL, TASKS


SOURCE_SHA256 = {
    "LICENSE": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
    "scripts/synthetic.yaml": "34bc71dcacdc41a829a170f04b528fbf48d62c616338005ab4991680fbf8cb0b",
    "scripts/data/template.py": "2e82d85152212136fffcd5624b158a2c75fd5036ec1c537f4b21eeb78fd18069",
    "scripts/data/tokenizer.py": "c2e4bfab607eef87a86334558303c1811bc8e93f22a5c4b129f302726d2357a4",
    "scripts/data/manifest_utils.py": "ecac79322f28ce9a12388b12e35d560ecb7cdad6b9888467a5a13b0eff2db91e",
    "scripts/data/synthetic/constants.py": "6296e901d495ec6200dc3f68993ea13d8282e3c0dbe1a8c47967f111105d1fde",
    "scripts/data/synthetic/niah.py": "53f2e2e0c61607d3814303c7f22b2f419fd004eb3bfe92a00ecf811e31aa8451",
}
DEPENDENCIES = {
    "nltk": "3.9.2",
    "numpy": "2.2.6",
    "PyYAML": "6.0.2",
    "tenacity": "9.1.2",
    "tiktoken": "0.12.0",
    "tqdm": "4.67.1",
    "wonderwords": "2.2.0",
}
NLTK_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_URL = (
    f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_REVISION}"
    "/packages/tokenizers/punkt_tab.zip"
)
NLTK_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
NLTK_MEMBERS = (
    "punkt_tab/README",
    "punkt_tab/english/abbrev_types.txt",
    "punkt_tab/english/collocations.tab",
    "punkt_tab/english/ortho_context.tab",
    "punkt_tab/english/sent_starters.txt",
)
TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
ESSAY_SEPARATOR = "\n\n"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _download(url: str, expected_sha256: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LocalLLMBench-RULER-preparation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Integrity check failed for {url}: expected {expected_sha256}, got {digest}")
    return data


def _dependency_versions() -> dict:
    versions = {}
    for package, expected in DEPENDENCIES.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"Preparation requires {package}=={expected}, found {actual}; rebuild the benchmark image")
        versions[package] = actual
    # Record supporting packages too: changing the preparation environment must
    # be visible even when the direct generator dependencies remain unchanged.
    for package in ("click", "joblib", "regex", "requests", "certifi", "charset-normalizer", "idna", "urllib3"):
        versions[package] = importlib.metadata.version(package)
    return versions


def _copy_essays(essays: Path, source: Path) -> dict:
    if not essays.is_dir():
        raise ValueError(f"Essay directory does not exist: {essays}")
    files = sorted((path for path in essays.iterdir() if path.is_file() and path.suffix.lower() == ".txt"), key=lambda path: path.name)
    if not files:
        raise ValueError(f"No UTF-8 .txt essays found in {essays}")
    texts = []
    identities = []
    for path in files:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Essay is empty: {path.name}")
        texts.append(text)
        identities.append({"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    text = ESSAY_SEPARATOR.join(texts)
    destination = source / "scripts/data/synthetic/json/PaulGrahamEssays.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_json_bytes({"text": text}))
    return {
        "source": "local_user_supplied_essays",
        "files": identities,
        "encoding": "utf-8",
        "order": "filename_lexicographic",
        "separator": ESSAY_SEPARATOR,
        "concatenated_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalization": "none_before_unmodified_upstream_generator",
    }


def _resources(work: Path) -> tuple[Path, dict, dict]:
    source = work / "upstream"
    for relative, digest in SOURCE_SHA256.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/NVIDIA/RULER/{GENERATOR_REVISION}/{relative}"
        destination.write_bytes(_download(url, digest))

    nltk_root = work / "nltk_data"
    member_hashes = {}
    with zipfile.ZipFile(io.BytesIO(_download(NLTK_URL, NLTK_SHA256))) as archive:
        # Extract only the fixed English text tables, never pickle resources or
        # arbitrary archive paths. NLTK 3.9.2's sent_tokenize uses punkt_tab.
        for member in NLTK_MEMBERS:
            data = archive.read(member)
            destination = nltk_root / "tokenizers" / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            member_hashes[member] = hashlib.sha256(data).hexdigest()

    cache = work / "tiktoken_cache"
    cache.mkdir()
    # tiktoken's documented loader cache keys are SHA1(URL); the downloaded
    # contents are independently authenticated with the upstream SHA256.
    cache_key = hashlib.sha1(TOKENIZER_URL.encode("utf-8")).hexdigest()
    (cache / cache_key).write_bytes(_download(TOKENIZER_URL, TOKENIZER_SHA256))
    environment = os.environ.copy()
    environment.update({
        "NLTK_DATA": str(nltk_root),
        "TIKTOKEN_CACHE_DIR": str(cache),
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    resource_identity = {
        "repository": "https://github.com/nltk/nltk_data",
        "revision": NLTK_REVISION,
        "url": NLTK_URL,
        "sha256": NLTK_SHA256,
        "files_sha256": member_hashes,
        "language": "english",
        "resource": "punkt_tab",
    }
    return source, environment, resource_identity


def _encoding(cache: str):
    import tiktoken

    previous = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = cache
    try:
        return tiktoken.get_encoding("cl100k_base")
    finally:
        if previous is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = previous


def _records(path: Path, task: str, count: int, budget: int, reserved: int, encoding) -> list[dict]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        if not isinstance(record, dict) or type(record.get("index")) is not int or record["index"] != index:
            raise ValueError(f"{task}: invalid or duplicate upstream sample index {index}")
        for field in ("input", "answer_prefix"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{task}/{index}: missing {field}")
        references = record.get("outputs")
        if not isinstance(references, list) or not references or not all(isinstance(value, str) and value.strip() for value in references):
            raise ValueError(f"{task}/{index}: invalid references")
        text = record["input"] + record["answer_prefix"]
        tokens = len(encoding.encode(text))
        if not 0 < tokens <= budget:
            raise ValueError(f"{task}/{index}: actual reference input tokens {tokens} exceed budget {budget}")
        if record.get("length") != tokens + reserved:
            raise ValueError(f"{task}/{index}: upstream token length disagrees with reference tokenizer")
        if not all(value in record["input"] for value in references):
            raise ValueError(f"{task}/{index}: reference answer is absent from the generated haystack")
        records.append({**record, "id": f"{task}/{index}", "task": task, "input_tokens": tokens})
    if len(records) != count:
        raise ValueError(f"{task}: expected {count} samples, received {len(records)}")
    return records


def prepare_corpus(args: argparse.Namespace) -> dict:
    """Generate and validate every requested variant before publishing files."""
    import yaml

    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing prepared output: {output}. Choose a new output directory.")
    if not args.essays.is_dir():
        raise ValueError(f"Essay directory does not exist: {args.essays}")
    dependencies = _dependency_versions()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ruler-prepare-", dir=output.parent) as temporary:
        work = Path(temporary)
        source, environment, sentence_tokenizer = _resources(work)
        essays = _copy_essays(args.essays, source)
        encoding = _encoding(environment["TIKTOKEN_CACHE_DIR"])
        configurations = yaml.safe_load((source / "scripts/synthetic.yaml").read_text(encoding="utf-8"))
        base_task = runpy.run_path(str(source / "scripts/data/synthetic/constants.py"))["TASKS"]["niah"]
        templates = runpy.run_path(str(source / "scripts/data/template.py"))["Templates"]
        template = templates["base"].format(task_template=base_task["template"]) + base_task["answer_prefix"]
        generated = work / "generated"
        corpus = work / "corpus.jsonl"
        digest = hashlib.sha256()
        with corpus.open("wb") as stream:
            for task in args.tasks:
                configuration = configurations[task]
                if configuration["task"] != "niah":
                    raise ValueError(f"Unexpected upstream generator for {task}")
                command = [
                    sys.executable, str(source / "scripts/data/synthetic/niah.py"),
                    "--save_dir", str(generated), "--save_name", task,
                    "--subset", "validation", "--tokenizer_path", "cl100k_base",
                    "--tokenizer_type", "openai",
                    "--max_seq_length", str(args.input_tokens + args.reserved_output_tokens),
                    "--tokens_to_generate", str(args.reserved_output_tokens),
                    "--num_samples", str(args.samples_per_task),
                    "--random_seed", str(args.seed), "--template", template,
                ]
                for key, value in configuration["args"].items():
                    command.extend((f"--{key}", str(value)))
                print(f"Preparing {task}: {args.samples_per_task} samples", flush=True)
                subprocess.run(command, cwd=source, env=environment, check=True)
                for record in _records(generated / task / "validation.jsonl", task, args.samples_per_task, args.input_tokens, args.reserved_output_tokens, encoding):
                    data = _json_bytes(record)
                    stream.write(data)
                    digest.update(data)
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "corpus_sha256": digest.hexdigest(),
            "generator": {
                "repository": GENERATOR_REPOSITORY,
                "revision": GENERATOR_REVISION,
                "files_sha256": SOURCE_SHA256,
                "entrypoint": "scripts/data/synthetic/niah.py",
                "model_template": "base",
                "source_modified": False,
            },
            "tokenizer": {
                "encoding": "cl100k_base", "package": "tiktoken",
                "version": dependencies["tiktoken"],
                "url": TOKENIZER_URL, "sha256": TOKENIZER_SHA256,
                "counting_policy": "input_plus_answer_prefix_without_chat_wrapper",
                "model_native": False,
            },
            "sentence_tokenizer": sentence_tokenizer,
            "generation": {
                "seed": args.seed, "input_tokens": args.input_tokens,
                "reserved_output_tokens": args.reserved_output_tokens,
                "samples_per_task": args.samples_per_task, "tasks": list(args.tasks),
            },
            "essays": essays,
            "dependencies": dependencies,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "adaptations": [
                "Local user-supplied essays, not the original upstream essay corpus",
                "Fixed cl100k_base reference input budget, not model-native tokens",
                "Base task template and answer prefix form one user chat message",
                "Reserved output budget replaces the upstream 128-token default",
                "Eight retrieval variants only; not the full RULER benchmark",
            ],
        }
        (work / "manifest.json").write_bytes(_json_bytes(manifest))
        # Exclusive creation also guards against another preparation publishing
        # to the same destination while the generator was running.
        output.mkdir()
        for name in ("corpus.jsonl", "manifest.json"):
            (work / name).rename(output / name)
        (output / "THIRD_PARTY_NOTICES.txt").write_bytes(Path(__file__).with_name("THIRD_PARTY_NOTICES.txt").read_bytes())
        (output / "punkt_tab_README.txt").write_bytes((work / "nltk_data/tokenizers/punkt_tab/README").read_bytes())
    return manifest


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--essays", type=Path, required=True, help="Read-only directory of local UTF-8 .txt essays")
    parser.add_argument("--output", type=Path, required=True, help="New immutable prepared directory; must not exist")
    parser.add_argument("--input-tokens", type=_positive_integer, default=65536, help="cl100k_base input ceiling (minimum 4096)")
    parser.add_argument("--samples-per-task", type=_positive_integer, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reserved-output-tokens", type=_positive_integer, default=8192)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    args = parser.parse_args()
    if args.input_tokens < 4096:
        parser.error("--input-tokens must be at least 4096 for the unmodified upstream generator")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1 for the upstream NumPy seed")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    # Artifact task order is always canonical, independent of CLI argument order.
    args.tasks = [task for task in TASKS if task in args.tasks]
    try:
        manifest = prepare_corpus(args)
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(1, f"RULER preparation failed: {error}\n")
    print(f"Prepared {len(args.tasks) * args.samples_per_task} samples; corpus SHA256 {manifest['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
