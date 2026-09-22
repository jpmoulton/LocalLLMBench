"""Evaluator image preparation: pure plans, context staging, scripted Docker preparation and the lock script.
No Docker, network, GPU or model is touched; every argv passes the real preparation vocabulary."""

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

from llmbench.coding.sandbox import WorkerResult
from llmbench.containers.capabilities import help_sha256
from llmbench.containers.config import ImageBundle, read_image_bundle
from llmbench.containers.executor import ComposeExecutor
from llmbench.containers.image_plan import (PrepareInputs, build_evaluator_context, build_wheel, evaluator_image_build_plan,
                                            lock_is_hash_pinned, prepare_images, read_iidfile, source_dir)
from llmbench.registry import builtin_registry
from llmbench.safety import OperationForbidden, SessionLock

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
HELP = (DATA / "llama-server-help-b11011.txt").read_bytes()
VERSION = (DATA / "llama-server-version-b11011.txt").read_bytes()
BASE = "python:3.12-slim-bookworm@sha256:" + "a" * 64
INFERENCE = "ghcr.io/ggml-org/llama.cpp@sha256:" + "d" * 64
INFERENCE_ID, EVALUATOR_ID, WORKER_ID = ("sha256:" + c * 64 for c in "fed")
OK = WorkerResult("completed", 0, b"ok")
LOCK = ("# derived\n" "pydantic==2.13.5 --hash=sha256:" + "1" * 64 + "\n" "inspect_ai==0.3.265 --hash=sha256:"
        + "2" * 64 + "\n")


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    # Preparation discovers staged datasets relative to cwd; never consume the developer's live corpora.
    monkeypatch.chdir(tmp_path)


def wheel_file(directory: Path, name="localllmbench-0.1.0-py3-none-any.whl") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"PK\x03\x04wheel-bytes")
    return path


def lock_file(directory: Path, text=LOCK) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "requirements.linux.lock"
    path.write_bytes(text.encode("utf-8"))  # exact bytes: write_text would emit CRLF on Windows
    return path


def self_check_report(**changes) -> dict:
    return {"ok": True, "errors": [], "registry_digest": builtin_registry().digest(), "tasks": {},
            "python_version": "Python 3.12.14", "pip_freeze_sha256": "9" * 64, **changes}


class ScriptedPrep:
    """Scripted preparation executor; every argv is validated against the real bounded vocabulary."""

    def __init__(self, *, report=None, entrypoint=("python", "-m", "llmbench.container_eval"), user="10001:10001",
                 cmd=None, build=OK, self_check_result=None):
        self.calls, self.validate = [], ComposeExecutor(allow_preparation=True)._validate
        self.report = self_check_report() if report is None else report
        self.entrypoint, self.user, self.cmd, self.build, self.self_check_result = entrypoint, user, cmd, build, self_check_result

    def run(self, argv, *, timeout_seconds, max_output_bytes):
        self.validate(argv)
        assert type(timeout_seconds) is int and timeout_seconds >= 1 and max_output_bytes >= 1
        self.calls.append(argv)
        if argv[1] == "pull":
            return OK
        if argv[1:3] == ("image", "inspect"):
            reference = argv[-1]
            if reference == INFERENCE:
                raw = {"Id": INFERENCE_ID, "RepoDigests": [INFERENCE], "Os": "linux", "Architecture": "amd64",
                       "Config": {"Entrypoint": ["/app/llama-server"], "Env": ["LLAMA_ARG_HOST=0.0.0.0"]}}
            elif reference == EVALUATOR_ID:
                raw = {"Id": EVALUATOR_ID, "RepoDigests": [], "Os": "linux", "Architecture": "amd64",
                       "Config": {"Entrypoint": list(self.entrypoint), "Cmd": self.cmd, "User": self.user, "Env": []}}
            elif reference == WORKER_ID:
                raw = {"Id": WORKER_ID, "RepoDigests": [], "Os": "linux", "Architecture": "amd64",
                       "Config": {"Entrypoint": ["python3"], "Cmd": ["--version"], "User": "10001:10001"}}
            else:
                return WorkerResult("completed", 1, b"", b"No such image")
            return WorkerResult("completed", 0, json.dumps(raw).encode())
        if argv[1] == "build":
            Path(argv[6]).write_text(EVALUATOR_ID, encoding="ascii")
            return self.build
        if argv[-1] == "--help":
            return WorkerResult("completed", 0, HELP)
        if argv[-1] == "--version":
            return WorkerResult("completed", 0, VERSION)
        if argv[-1] == "--self-check":
            return self.self_check_result or WorkerResult("completed", 0, json.dumps(self.report).encode())
        raise AssertionError(f"unexpected argv {argv}")


def inputs(tmp_path, **changes):
    values = {"inference": INFERENCE, "evaluator_base": BASE, "wheel_path": wheel_file(tmp_path / "wheel"),
              "lock_path": lock_file(tmp_path / "lock")}
    values.update(changes)
    return PrepareInputs(**values)


def test_evaluator_build_plan_requires_digest_base_and_records_hashes(tmp_path):
    wheel, lock = wheel_file(tmp_path / "w"), lock_file(tmp_path / "l")
    plan = evaluator_image_build_plan(base_image=BASE, artifact_dir=tmp_path / "out", wheel_path=wheel, lock_path=lock)
    assert plan["status"] == "planned-not-executed" and plan["platform"] == "linux/amd64"
    assert plan["build_argv"] == ["docker", "build", "--platform=linux/amd64", "--pull=false", "--no-cache",
                                  "--iidfile", str((tmp_path / "out").resolve() / "evaluator-image.id"), "--file",
                                  str((tmp_path / "out").resolve() / "evaluator-context" / "Dockerfile"),
                                  str((tmp_path / "out").resolve() / "evaluator-context")]
    ComposeExecutor(allow_preparation=True)._validate(tuple(plan["build_argv"]))
    source = source_dir()
    assert plan["source_sha256"] == {
        "Dockerfile": hashlib.sha256((source / "Dockerfile").read_bytes()).hexdigest(),
        ".dockerignore": hashlib.sha256((source / ".dockerignore").read_bytes()).hexdigest(),
        "requirements.linux.lock": hashlib.sha256(LOCK.encode()).hexdigest(),
        wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()}
    assert plan["runtime"]["entrypoint"] == ["python", "-m", "llmbench.container_eval"] and plan["runtime"]["cmd"] is None
    # The Dockerfile installs the lock with --require-hashes but no --no-index: every wheel is downloaded from PyPI
    # at build time (hash-verified). The recorded provenance must say so (REV-A1-04); the base is pulled before.
    assert plan["build_requires_network"] is True and "--tag" not in plan["build_argv"]
    assert plan["build_downloads"] and all(isinstance(item, str) for item in plan["build_downloads"])
    assert any("requirements.linux.lock" in item and "PyPI" in item for item in plan["build_downloads"])
    assert "--pull=false" in plan["build_argv"] and plan["base_image_pulled_before_build"] is True
    for bad in ("python:3.12-slim-bookworm", "python:3.12-slim-bookworm@sha256:abc", "python:latest@sha256:" + "a" * 64,
                "node:24-bookworm-slim@sha256:" + "a" * 64, "evil.example/python:3.12-slim-bookworm@sha256:" + "a" * 64,
                None, 5):
        with pytest.raises(ValueError, match="digest"):
            evaluator_image_build_plan(base_image=bad, artifact_dir=tmp_path, wheel_path=wheel, lock_path=lock)
    unhashed = lock_file(tmp_path / "u", "pydantic==2.13.5\n")
    with pytest.raises(ValueError, match="hash-pinned"):
        evaluator_image_build_plan(base_image=BASE, artifact_dir=tmp_path, wheel_path=wheel, lock_path=unhashed)
    foreign = wheel_file(tmp_path / "f", "other_project-1.0-py3-none-any.whl")
    with pytest.raises(ValueError, match="localllmbench"):
        evaluator_image_build_plan(base_image=BASE, artifact_dir=tmp_path, wheel_path=foreign, lock_path=lock)
    with pytest.raises((ValueError, OSError)):
        evaluator_image_build_plan(base_image=BASE, artifact_dir=tmp_path, wheel_path=tmp_path / "missing.whl",
                                   lock_path=lock)
    assert lock_is_hash_pinned("pydantic==2.13.5 --hash=sha256:" + "1" * 64 + " --hash=sha256:" + "2" * 64) == (
        False, ["line 1: not a `name==version --hash=sha256:<hex>` pin"])


def test_context_staging_copies_exact_files_and_rejects_symlinks(tmp_path):
    wheel, lock = wheel_file(tmp_path / "w"), lock_file(tmp_path / "l")
    staging = tmp_path / "ctx"
    digests = build_evaluator_context(staging, wheel_path=wheel, base_image=BASE, lock_path=lock)
    assert sorted(path.name for path in staging.iterdir()) == sorted([".dockerignore", "Dockerfile",
                                                                       "requirements.linux.lock", wheel.name,
                                                                       "benchmarks"])
    rendered = (staging / "Dockerfile").read_text(encoding="utf-8")
    assert f"\nARG BASE_IMAGE={BASE}\n" in rendered and rendered.count("ARG BASE_IMAGE") == 1
    assert "\nARG BASE_IMAGE\n" not in rendered
    source = (source_dir() / "Dockerfile").read_text(encoding="utf-8")
    assert rendered.replace(f"ARG BASE_IMAGE={BASE}", "ARG BASE_IMAGE") == source
    assert digests == {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in digests}
    assert (staging / "requirements.linux.lock").read_text(encoding="utf-8") == LOCK
    assert (staging / ".dockerignore").read_bytes() == (source_dir() / ".dockerignore").read_bytes()
    assert (staging / "benchmarks").is_dir()
    with pytest.raises(ValueError, match="new or empty"):
        build_evaluator_context(staging, wheel_path=wheel, base_image=BASE, lock_path=lock)
    with pytest.raises(ValueError, match="digest"):
        build_evaluator_context(tmp_path / "ctx2", wheel_path=wheel, base_image="python:3.12-slim-bookworm", lock_path=lock)
    with pytest.raises(ValueError, match="hash-pinned"):
        build_evaluator_context(tmp_path / "ctx3", wheel_path=wheel, base_image=BASE,
                                lock_path=lock_file(tmp_path / "u", "pydantic==2.13.5\n"))
    assert not (tmp_path / "ctx3").exists()
    try:
        os.symlink(wheel, tmp_path / "linked.whl")
        os.symlink(tmp_path / "elsewhere", tmp_path / "linked-staging", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable to this account")
    with pytest.raises(ValueError, match="regular file"):
        build_evaluator_context(tmp_path / "ctx4", wheel_path=tmp_path / "linked.whl", base_image=BASE, lock_path=lock)
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="never a link"):
        build_evaluator_context(tmp_path / "linked-staging", wheel_path=wheel, base_image=BASE, lock_path=lock)
    assert not list((tmp_path / "elsewhere").iterdir())


def test_context_staging_includes_nested_benchmark_data(tmp_path):
    wheel, lock = wheel_file(tmp_path / "w"), lock_file(tmp_path / "l")
    datasets = tmp_path / "datasets"
    corpus = datasets / "ruler" / "essays"
    corpus.mkdir(parents=True)
    (corpus / "sample.txt").write_bytes(b"retrieval corpus\n")
    (datasets / "staging-manifest.json").write_bytes(b'{"schema_version": 1}\n')
    staging = tmp_path / "ctx"
    digests = build_evaluator_context(staging, wheel_path=wheel, base_image=BASE, lock_path=lock,
                                      dataset_root=datasets)
    for relative in ("ruler/essays/sample.txt", "staging-manifest.json"):
        expected = (datasets / relative).read_bytes()
        assert (staging / "benchmarks" / relative).read_bytes() == expected
        assert digests[f"benchmarks/{relative}"] == hashlib.sha256(expected).hexdigest()


def test_prepare_images_records_bundle_from_scripted_docker(tmp_path):
    docker = ScriptedPrep()
    worker_iidfile = tmp_path / "worker-image.id"
    worker_iidfile.write_text(WORKER_ID + "\n", encoding="ascii")
    clock = iter(float(index) for index in range(1000))
    output = tmp_path / "prep"
    bundle = prepare_images(inputs(tmp_path, worker_iidfile=worker_iidfile), executor=docker, artifact_dir=output,
                            clock=lambda: next(clock), session_lock=SessionLock(True, True, True))
    kinds = [(argv[1], argv[-1]) for argv in docker.calls]
    # The base is pulled as `python@sha256:<digest>`: the tag is display only and the executor never accepts tags.
    assert kinds == [("pull", INFERENCE), ("image", INFERENCE), ("run", "--help"), ("run", "--version"),
                     ("pull", "python@sha256:" + "a" * 64), ("build", str(output.resolve() / "evaluator-context")),
                     ("image", EVALUATOR_ID), ("run", "--self-check"), ("image", WORKER_ID)]
    self_check = docker.calls[7]
    assert self_check[2:8] == ("--rm", "--pull=never", "--network=none", "--read-only", "--user=10001:10001", EVALUATOR_ID)
    assert isinstance(bundle, ImageBundle) and bundle.inference.image_id == INFERENCE_ID
    assert bundle.inference.help_sha256 == help_sha256(HELP.decode("utf-8")) == bundle.help_sha256
    assert bundle.inference.build_info == "b11011-aa39d7a3e" and bundle.inference.entrypoint == ("/app/llama-server",)
    assert bundle.evaluator.image_id == bundle.evaluator.reference == EVALUATOR_ID
    assert bundle.evaluator.entrypoint == ("python", "-m", "llmbench.container_eval")
    assert bundle.worker.image_id == WORKER_ID and bundle.worker.role == "worker"
    assert bundle.registry_digest == builtin_registry().digest()
    build = bundle.evaluator_build
    assert build["plan"]["base_image"] == BASE and build["self_check"]["ok"] is True
    assert set(build["staged_sha256"]) == {"Dockerfile", ".dockerignore", "requirements.linux.lock",
                                           "localllmbench-0.1.0-py3-none-any.whl"}
    assert [entry["argv"][1] for entry in build["docker_log"]] == ["pull", "run", "run", "pull", "build", "run"]
    saved = read_image_bundle(output / "image-bundle.json")
    assert saved == bundle
    assert (output / "llama-server-help.txt").read_bytes() == HELP
    assert (output / "llama-server-version.txt").read_bytes() == VERSION
    assert json.loads((output / "evaluator-self-check.json").read_text())["ok"] is True
    assert read_iidfile(output / "evaluator-image.id") == EVALUATOR_ID
    assert (output / "evaluator-context" / "Dockerfile").exists() and not (output / "image-bundle.json.tmp").exists()
    with pytest.raises(ValueError, match="already exists"):
        prepare_images(inputs(tmp_path), executor=docker, artifact_dir=output)
    with pytest.raises(OperationForbidden):
        prepare_images(inputs(tmp_path), executor=docker, artifact_dir=tmp_path / "denied", session_lock=SessionLock())
    assert not (tmp_path / "denied").exists()


@pytest.mark.parametrize("options, fragment", [
    ({"report": self_check_report(ok=False, errors=["import inspect_ai: ModuleNotFoundError"])}, "ok=true"),
    ({"report": self_check_report(registry_digest="0" * 64)}, "registry digest differs"),
    ({"report": self_check_report(python_version=None)}, "provenance"),
    ({"self_check_result": WorkerResult("completed", 3, b"", b"traceback")}, "self-check failed"),
    ({"self_check_result": WorkerResult("completed", 0, b"not json")}, "not JSON"),
    ({"entrypoint": ("python",)}, "entrypoint"), ({"cmd": ["--version"]}, "no CMD"), ({"user": "0:0"}, "10001:10001"),
    ({"build": WorkerResult("timeout", None, b"", b"stalled")}, "docker build"),
])
def test_prepare_refuses_when_self_check_fails_or_registry_digest_differs(tmp_path, options, fragment):
    docker = ScriptedPrep(**options)
    output = tmp_path / "prep"
    with pytest.raises(ValueError, match=fragment):
        prepare_images(inputs(tmp_path), executor=docker, artifact_dir=output)
    assert not (output / "image-bundle.json").exists() and not (output / "image-bundle.json.tmp").exists()
    for argv in docker.calls:
        assert "--privileged" not in argv and "--network=host" not in argv


def test_build_wheel_replaces_same_version_only_after_successful_build(tmp_path):
    output = tmp_path / "wheels"
    stale = wheel_file(output)
    stale.write_bytes(b"previous build")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "-c":
            return OK
        target = Path(argv[argv.index("--wheel-dir") + 1])
        assert target != output and not list(target.iterdir())
        (target / stale.name).write_bytes(b"fresh build")
        return OK

    result = build_wheel(tmp_path, output, python="chosen-python", run=run)
    assert result == stale and result.read_bytes() == b"fresh build"
    assert all(argv[0] == "chosen-python" for argv in calls)
    assert list(output.iterdir()) == [result]


@pytest.mark.parametrize("result, fragment", [
    (WorkerResult("completed", 1, b"", b"backend failed"), "pip wheel failed"),
    (OK, "exactly one newly built wheel"),
])
def test_build_wheel_failed_retry_never_returns_stale_wheel(tmp_path, result, fragment):
    output = tmp_path / "wheels"
    stale = wheel_file(output)
    before = stale.read_bytes()

    def run(argv, **kwargs):
        if argv[1] == "-c":
            return OK
        if result.returncode != 0:
            target = Path(argv[argv.index("--wheel-dir") + 1])
            (target / stale.name).write_bytes(b"incomplete build")
        return result

    with pytest.raises(ValueError, match=fragment):
        build_wheel(tmp_path, output, run=run)
    assert stale.read_bytes() == before
    assert list(output.iterdir()) == [stale]


def test_build_wheel_missing_backend_explains_chosen_interpreter_install(tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return WorkerResult("completed", 1, b"", b"ModuleNotFoundError: setuptools")

    with pytest.raises(ValueError) as caught:
        build_wheel(tmp_path, tmp_path / "wheels", python="chosen-python", run=run)
    assert '"chosen-python" -m pip install --upgrade pip setuptools wheel' in str(caught.value)
    assert '"chosen-python" -m ensurepip --upgrade' in str(caught.value)
    assert len(calls) == 1 and calls[0][1] == "-c"
    assert not (tmp_path / "wheels").exists()


def test_prepare_retry_preserves_old_context_and_unrelated_paths(tmp_path):
    output = tmp_path / "prep"
    prepared_inputs = inputs(tmp_path)
    failed = ScriptedPrep(build=WorkerResult("completed", 1, b"", b"failed build"))
    with pytest.raises(ValueError, match="docker build"):
        prepare_images(prepared_inputs, executor=failed, artifact_dir=output)
    old_context = output / "evaluator-context"
    unrelated = old_context / "keep-user-data.txt"
    unrelated.write_bytes(b"keep context contents")
    sibling = output / "unrelated"
    sibling.mkdir()
    (sibling / "notes.txt").write_bytes(b"keep sibling contents")
    old_atomic_temporary = output / "image-bundle.json.tmp"
    old_atomic_temporary.write_bytes(b"interrupted old write")
    old_iid = (output / "evaluator-image.id").read_bytes()
    prepared_inputs.wheel_path.write_bytes(b"updated wheel")

    bundle = prepare_images(prepared_inputs, executor=ScriptedPrep(), artifact_dir=output)
    plan = bundle.evaluator_build["plan"]
    context = Path(plan["context"])
    assert context != old_context
    assert (context / prepared_inputs.wheel_path.name).read_bytes() == b"updated wheel"
    assert unrelated.read_bytes() == b"keep context contents"
    assert (sibling / "notes.txt").read_bytes() == b"keep sibling contents"
    assert old_atomic_temporary.read_bytes() == b"interrupted old write"
    assert (output / "evaluator-image.id").read_bytes() == old_iid
    assert Path(plan["iidfile"]) != output / "evaluator-image.id"
    assert read_iidfile(plan["iidfile"]) == EVALUATOR_ID
    assert read_image_bundle(output / "image-bundle.json") == bundle
    refused = ScriptedPrep()
    with pytest.raises(ValueError, match="already exists"):
        prepare_images(prepared_inputs, executor=refused, artifact_dir=output)
    assert not refused.calls


def test_prepare_retry_after_interrupted_context_staging(tmp_path, monkeypatch):
    import llmbench.containers.image_plan as image_plan

    output = tmp_path / "prep"
    old_context = output / "evaluator-context"
    old_context.mkdir(parents=True)
    (old_context / "unrelated.txt").write_bytes(b"do not delete")
    prepared_inputs = inputs(tmp_path)
    failed = ScriptedPrep()

    def interrupted(staging, **kwargs):
        Path(staging).mkdir()
        (Path(staging) / "partial").write_bytes(b"incomplete")
        raise OSError("interrupted staging")

    with monkeypatch.context() as patch:
        patch.setattr(image_plan, "build_evaluator_context", interrupted)
        with pytest.raises(OSError, match="interrupted staging"):
            prepare_images(prepared_inputs, executor=failed, artifact_dir=output)
    assert not failed.calls
    assert sorted(path.name for path in output.iterdir()) == ["evaluator-context"]
    datasets = tmp_path / "artifacts" / "benchmark-datasets" / "ruler"
    datasets.mkdir(parents=True)
    (datasets / "sample.txt").write_bytes(b"staged corpus")
    bundle = prepare_images(prepared_inputs, executor=ScriptedPrep(), artifact_dir=output)
    context = Path(bundle.evaluator_build["plan"]["context"])
    assert context != old_context
    assert (old_context / "unrelated.txt").read_bytes() == b"do not delete"
    assert (context / "benchmarks" / "ruler" / "sample.txt").read_bytes() == b"staged corpus"
    assert bundle.evaluator_build["staged_sha256"]["benchmarks/ruler/sample.txt"] == hashlib.sha256(
        b"staged corpus").hexdigest()


def test_lock_script_output_is_hash_pinned_and_excludes_host_only_packages(tmp_path):
    spec = importlib.util.spec_from_file_location("lock_evaluator_requirements",
                                                  ROOT / "scripts" / "lock_evaluator_requirements.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = ROOT / "requirements.lock.txt"
    pins = module.derive_pins(source.read_text(encoding="utf-8"))
    names = {module.normalize(name) for name, _ in pins}
    assert {"inspect-ai", "pydantic", "httpx", "numpy"} <= names
    assert not names & {"lmstudio", "pytest", "ruff", "debugpy"}
    assert ("inspect_ai", "0.3.265") in pins and ("pydantic", "2.13.5") in pins
    with pytest.raises(ValueError, match="exact"):
        module.derive_pins("pydantic>=2\n")
    with pytest.raises(ValueError, match="no pins"):
        module.derive_pins("# empty\n")
    downloads = []

    def fake_pip(argv, **kwargs):
        downloads.append(argv)
        assert argv[1:4] == ["-m", "pip", "download"] and "--no-deps" in argv and "--only-binary=:all:" in argv
        # pip does not expand a PEP 600 tag given on the CLI: every platform the bookworm (glibc 2.36) base can
        # run must be passed explicitly, newest first, and the unvetted bare linux_x86_64 tag never (REV-A1-01).
        platforms = [argv[index + 1] for index, item in enumerate(argv) if item == "--platform"]
        assert platforms == ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"]
        assert "linux_x86_64" not in argv and "--implementation" in argv and argv[argv.index("--implementation") + 1] == "cp"
        assert argv[argv.index("--python-version") + 1] == "3.12"
        destination = Path(argv[argv.index("--dest") + 1])
        for line in Path(argv[argv.index("--requirement") + 1]).read_text(encoding="utf-8").splitlines():
            name, version = line.split("==")
            (destination / f"{name.replace('-', '_')}-{version}-py3-none-any.whl").write_bytes(line.encode())
        return WorkerResult("completed", 0)

    output = tmp_path / "requirements.linux.lock"
    text = module.build_lock(source, output, python="python", run=fake_pip, scratch=tmp_path)
    assert output.read_text(encoding="utf-8") == text and len(downloads) == 1
    pinned, problems = lock_is_hash_pinned(text)
    assert pinned and problems == []
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any(all(platform in line for platform in module.PLATFORMS) for line in header)
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert len(lines) == len(pins) and all(re.fullmatch(r"[^ ]+==[^ ]+ --hash=sha256:[0-9a-f]{64}", line) for line in lines)
    expected = hashlib.sha256(b"pydantic==2.13.5").hexdigest()
    assert f"pydantic==2.13.5 --hash=sha256:{expected}" in lines
    assert not any(line.startswith(("lmstudio==", "pytest==", "ruff==", "debugpy==")) for line in lines)
    assert not (tmp_path / "requirements.linux.lock.tmp").exists()

    def failing_pip(argv, **kwargs):
        return WorkerResult("completed", 1, b"", b"No matching distribution")
    with pytest.raises(ValueError, match="pip download failed"):
        module.build_lock(source, tmp_path / "other.lock", python="python", run=failing_pip, scratch=tmp_path)
    assert not (tmp_path / "other.lock").exists()

    def partial_pip(argv, **kwargs):
        Path(argv[argv.index("--dest") + 1]).joinpath("pydantic-2.13.5-py3-none-any.whl").write_bytes(b"x")
        return WorkerResult("completed", 0)
    with pytest.raises(ValueError, match="exactly one downloaded wheel"):
        module.build_lock(source, tmp_path / "partial.lock", python="python", run=partial_pip, scratch=tmp_path)


def test_dockerfile_pins_base_user_entrypoint_and_no_cmd():
    source = source_dir()
    text = (source / "Dockerfile").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert lines[0] == "ARG BASE_IMAGE" and lines[1] == "FROM ${BASE_IMAGE}"
    assert not any(line.startswith("CMD") for line in lines)
    assert lines[-1] == 'ENTRYPOINT ["python", "-m", "llmbench.container_eval"]'
    assert "USER 10001:10001" in lines and lines.index("USER 10001:10001") < len(lines) - 1
    assert any("--require-hashes" in line and "--no-deps" in line for line in lines)
    assert any("--no-index --no-deps" in line and ".whl" in line for line in lines)
    assert "PYTHONDONTWRITEBYTECODE=1" in text and "pip-freeze.txt" in text and "python-version.txt" in text
    # The comment must not claim an offline build: the lock install downloads (and hash-verifies) from PyPI.
    assert "Nothing is resolved at build time" not in text and "downloaded from PyPI" in text
    assert ":latest" not in text and "apt-get" not in text and "curl" not in text and "EXPOSE" not in text
    assert (source / ".dockerignore").read_text(encoding="utf-8").splitlines() == [
        "*", "!Dockerfile", "!requirements.linux.lock", "!*.whl", "!benchmarks", "!benchmarks/**"]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # The image sources (dotfile included) and the tune template must be packaged, or an installed wheel cannot
    # build the evaluator image or derive a session.
    packaged = next(line for line in pyproject.splitlines() if line.startswith('"llmbench.containers" ='))
    assert all(item in packaged for item in ('"evaluator_image/*"', '"evaluator_image/.dockerignore"', '"data/*.json"'))
