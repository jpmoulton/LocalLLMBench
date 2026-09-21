import json

import pytest
from pydantic import ValidationError

from llmbench.config import BackendSettings, Budgets, ExperimentManifest, RunMode, TaskSelection, canonical_json
from llmbench.safety import OperationForbidden, SessionLock
from llmbench.store import Store


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None, [], {}])
def test_permission_exact_bool(value):
    with pytest.raises(OperationForbidden):
        SessionLock(allow_inference=value)


@pytest.mark.parametrize("operation", ["load", "unload", "restore", "configure", "inference", "container"])
def test_session_default_denies(operation, tmp_path):
    with pytest.raises(OperationForbidden):
        SessionLock.read(tmp_path / "missing.json").check(operation, RunMode.LIVE)


def test_offline_denies_even_with_permission():
    with pytest.raises(OperationForbidden):
        SessionLock(allow_inference=True).check("inference", RunMode.OFFLINE)


def test_strict_config_roundtrip(manifest):
    other = ExperimentManifest.model_validate_json(manifest.model_dump_json())
    assert other.fingerprint() == manifest.fingerprint()
    data = json.loads(manifest.model_dump_json())
    data["backend"]["context_length"] = True
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate_json(json.dumps(data))
    data["backend"]["context_length"] = 8192
    data["backend"]["mtp"] = "false"
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate_json(json.dumps(data))


# LIVE-013: the fingerprint of the conftest manifest computed at 87f4965, before TaskSelection carried
# options. Pinned so the options field cannot perturb any manifest fingerprint already written to a ledger.
FINGERPRINT_BEFORE_OPTIONS = "bd54f274061d5983c6cd08320e827fb1b282f401f46e360f335dba96d633f3e3"


def test_manifest_without_options_keeps_its_pre_options_fingerprint(manifest):
    """A manifest declaring no benchmark options hashes exactly as it did before options existed."""
    assert manifest.tasks[0].options == {}
    assert manifest.fingerprint() == FINGERPRINT_BEFORE_OPTIONS
    # An explicitly empty mapping is the same experiment as no mapping at all.
    empty = manifest.model_copy(update={"tasks": (manifest.tasks[0].model_copy(update={"options": {}}),)})
    assert empty.fingerprint() == FINGERPRINT_BEFORE_OPTIONS
    # Manifest JSON recorded by a pre-options run still validates and still hashes to the same value.
    recorded = json.loads(manifest.model_dump_json())
    for task in recorded["tasks"]:
        del task["options"]
    assert ExperimentManifest.model_validate_json(json.dumps(recorded)).fingerprint() == FINGERPRINT_BEFORE_OPTIONS


def test_changed_benchmark_option_changes_the_manifest_fingerprint(manifest):
    """Options are measurement-affecting, so they are part of experimental identity (LIVE-013)."""
    base = manifest.tasks[0]

    def with_options(options):
        task = TaskSelection(suite=base.suite, revision=base.revision, task_ids=base.task_ids,
                             fixture_seed=base.fixture_seed, split=base.split, options=options)
        return ExperimentManifest.model_validate_json(canonical_json(
            {**json.loads(manifest.model_dump_json()), "tasks": [json.loads(task.model_dump_json())]}))

    capped = with_options({"ruler_output_tokens": 512})
    default = with_options({"ruler_output_tokens": 30})
    fingerprints = {manifest.fingerprint(), capped.fingerprint(), default.fingerprint(),
                    with_options({"ruler_output_tokens": 512, "lengths": [16384, 32768]}).fingerprint()}
    assert len(fingerprints) == 4
    # The value survives a JSON round-trip and so does the fingerprint.
    reread = ExperimentManifest.model_validate_json(capped.model_dump_json())
    assert reread.tasks[0].options == {"ruler_output_tokens": 512} and reread.fingerprint() == capped.fingerprint()


def test_task_selection_options_accept_exactly_what_benchmark_selection_accepts():
    """The two option carriers must not drift apart: a value one accepts the other must accept."""
    from llmbench.containers.config import BenchmarkSelection

    def task(options):
        return TaskSelection(suite="ruler", revision="r", task_ids=("ruler/vt/32768",), options=options)

    def benchmark(options):
        return BenchmarkSelection(benchmark_id="ruler", revision="r", task_ids=("ruler/vt/32768",), options=options)

    accepted = [{}, {"ruler_output_tokens": 512}, {"lengths": [16384, 131072]}, {"tasks": ["vt"]},
                {"strict": True}, {"ratio": 0.5}, {"name": "a" * 256}, {"lengths": list(range(64))},
                {f"k{index}": index for index in range(16)}]
    rejected = [{"bad key": 1}, {"": 1}, {"k" * 65: 1}, {"nested": {"a": 1}}, {"objects": [{"a": 1}]},
                {"too_long": "a" * 257}, {"nul": "a\x00b"}, {"lengths": list(range(65))},
                {f"k{index}": index for index in range(17)}]
    for options in accepted:
        assert dict(task(options).options) == dict(benchmark(options).options)
    for options in rejected:
        with pytest.raises(ValidationError):
            task(options)
        with pytest.raises(ValidationError):
            benchmark(options)


def test_task_selection_options_are_deeply_immutable():
    selection = TaskSelection(suite="ruler", revision="r", task_ids=("ruler/vt/32768",),
                              options={"ruler_output_tokens": 512, "lengths": [16384, 32768]})
    assert selection.options["lengths"] == (16384, 32768)  # lists freeze into tuples
    with pytest.raises(TypeError):
        selection.options["ruler_output_tokens"] = 30
    with pytest.raises(TypeError):
        selection.options.update({"ruler_output_tokens": 30})
    with pytest.raises(ValidationError):
        TaskSelection(suite="ruler", revision="r", task_ids=("ruler/vt/32768",), options=None)


def test_manifest_deep_immutable(manifest):
    original = manifest.fingerprint()
    with pytest.raises(TypeError):
        manifest.model.provenance["source"]["revisions"] += ("v2",)
    with pytest.raises(TypeError):
        manifest.annotations["label"] = "different"
    assert manifest.fingerprint() == original


def test_invalid_capacity(manifest):
    data = json.loads(manifest.model_dump_json())
    data["backend"]["context_length"] = 128
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate_json(json.dumps(data))


def test_reject_bool_threads_and_invalid_budget():
    with pytest.raises(ValidationError):
        BackendSettings(runtime_revision="x", context_length=1024, cpu_threads=True)
    with pytest.raises(ValidationError):
        Budgets(wall_seconds=10, reserve_validation_seconds=10)


def test_budgets_default_four_hours_and_llamacpp_engine_literal():
    assert Budgets().wall_seconds == 14400
    assert Budgets().reserve_validation_seconds < Budgets().wall_seconds
    backend = BackendSettings(engine="llama.cpp", runtime_revision="b11011-aa39d7a3e+0123456789ab",
                              context_length=8192, reasoning="off", ubatch_size=512)
    assert backend.engine == "llama.cpp" and backend.reasoning == "off" and backend.ubatch_size == 512
    assert BackendSettings(runtime_revision="x", context_length=1024).reasoning is None
    assert BackendSettings(runtime_revision="x", context_length=1024).ubatch_size is None
    with pytest.raises(ValidationError):
        BackendSettings(engine="vllm", runtime_revision="x", context_length=1024)
    with pytest.raises(ValidationError):
        BackendSettings(runtime_revision="x", context_length=1024, reasoning="maybe")
    with pytest.raises(ValidationError):
        BackendSettings(runtime_revision="x", context_length=1024, ubatch_size=16)


def test_store_attempts_preserve_failures(manifest, tmp_path):
    with Store(tmp_path) as store:
        first = store.create_attempt(manifest, synthetic=True)
        store.transition(first, "running")
        sample = {"task_id": "a", "status": "timeout", "score": 0}
        store.add_sample(first, sample)
        with pytest.raises(Exception):
            store.add_sample(first, sample)
        store.transition(first, "interrupted", "simulated process interruption")
        second = store.create_attempt(manifest, synthetic=True, parent_id=first)
        assert second != first
        assert store.results(first)["samples"] == [sample]
        with pytest.raises(ValueError):
            store.transition(first, "running")


def test_artifacts_immutable_safe_bounded(manifest, tmp_path):
    with Store(tmp_path, max_artifact_bytes=5) as store:
        attempt = store.create_attempt(manifest, synthetic=True)
        target = store.artifact(attempt, "raw.json", b"123")
        assert target.read_bytes() == b"123"
        with pytest.raises(ValueError):
            store.artifact(attempt, "too-big", b"123")
        for name in ("../escape", "C:alternate", "a\\b", ".."):
            with pytest.raises(ValueError):
                store.artifact(attempt, name, b"")
        with pytest.raises(FileExistsError):
            store.artifact(attempt, "raw.json", b"")


def test_exclusive_campaign_lock(tmp_path):
    with Store(tmp_path) as store:
        with store.campaign_lock():
            with pytest.raises(FileExistsError):
                with store.campaign_lock():
                    pass
        assert not (tmp_path / "campaign.lock").exists()


def test_trace_seals_partial_output_on_scoring_exception(manifest, tmp_path):
    with Store(tmp_path) as store:
        attempt = store.create_attempt(manifest, synthetic=True)
        store.transition(attempt, "running")
        with pytest.raises(RuntimeError):
            with store.trace(attempt) as write:
                write({"event": "message", "data": {"partial": "raw bytes before parser failure"}})
                raise RuntimeError("parser failed")
        trace = tmp_path / "raw" / attempt / "transport.jsonl"
        assert "raw bytes before parser failure" in trace.read_text()
        assert store.db.execute("SELECT count(*) FROM artifacts WHERE name='transport.jsonl'").fetchone()[0] == 1


def test_trace_and_artifacts_share_budget(manifest, tmp_path):
    with Store(tmp_path, max_artifact_bytes=30) as store:
        attempt = store.create_attempt(manifest, synthetic=True)
        store.transition(attempt, "running")
        with store.trace(attempt) as write:
            write({"a": 1})
            store.artifact(attempt, "extra", b"x" * 15)
            with pytest.raises(ValueError):
                write({"longer": 2})
