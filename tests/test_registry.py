import subprocess
import sys

import pytest

from llmbench.coding.adapters import SUITES
from llmbench.coding.fixtures import fixtures
from llmbench.containers.config import BenchmarkSelection
from llmbench.evaluations.tools import ToolEpisode, strict_argument_fixture
from llmbench.evaluations.retrieval import NIAH_VARIANTS
from llmbench.registry import Registry, RegistryEntry, RegistryError, builtin_registry


def select(benchmark_id, revision, *task_ids, **kwargs):
    return BenchmarkSelection(benchmark_id=benchmark_id, revision=revision, task_ids=task_ids, **kwargs)


def test_digest_is_stable_order_independent_and_content_sensitive():
    registry = builtin_registry()
    assert registry.digest() == builtin_registry().digest() == Registry(reversed(registry.entries)).digest()
    assert len(registry.digest()) == 64
    changed = [entry if entry.benchmark_id != "niah" else entry.model_copy(update={"revision": "local-niah-v2"})
               for entry in registry.entries]
    assert Registry(changed).digest() != registry.digest()
    with pytest.raises(ValueError):
        Registry([*registry.entries, registry.entries[0]])


def test_digest_is_identical_in_a_fresh_interpreter():
    code = "from llmbench.registry import builtin_registry; print(builtin_registry().digest())"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    assert output.stdout.strip() == builtin_registry().digest()


def test_runner_ids_map_one_to_one_onto_existing_suites_and_tasks():
    registry = builtin_registry()
    runners = {entry.benchmark_id: entry for entry in registry.entries if entry.capability == "runner"}
    local = {name: entry for name, entry in runners.items() if entry.task_namespace is None}
    assert set(local) == {"tool-probes", "tool-episodes", "niah"}
    assert set(local["tool-probes"].task_ids) == {strict_argument_fixture().task_id, "tools/no-call-v1"}
    assert local["tool-episodes"].task_ids == (ToolEpisode().task_id,)
    assert local["niah"].task_ids == tuple(NIAH_VARIANTS)
    assert all(not entry.requires and not entry.official for entry in local.values())
    # Dataset-backed public suites enumerate nothing here: their tasks live in the baked image.
    dataset_backed = {name: entry for name, entry in runners.items() if entry.task_namespace is not None}
    assert set(dataset_backed) == {"bfcl", "ruler", "evalplus", "aider-polyglot"}
    for name, entry in dataset_backed.items():
        assert entry.official is True and entry.task_ids == ()
        assert f"dataset:{name}" in entry.requires
    selection = select("niah", "local-niah-v1", "multi", split="holdout", seed=7).to_task_selection()
    assert (selection.suite, selection.revision, selection.fixture_seed, selection.split) == (
        "niah", "local-niah-v1", 7, "holdout")


def test_valid_runner_selections_pass_and_match_the_live_validator(manifest):
    selections = [select("tool-probes", "local-tools-v1", "tools/nested-exact-v1", "tools/no-call-v1"),
                  select("tool-episodes", "local-tools-v1", "tools/read-edit-test-v1"),
                  select("niah", "local-niah-v1", "single-middle", "multi"),
                  select("niah", "local-niah-v1", "multi", split="holdout", seed=7)]
    builtin_registry().validate(selections, broker_available=False)
    # every selection must also convert into the manifest's task form
    assert len(tuple(item.to_task_selection() for item in selections)) == len(selections)


@pytest.mark.parametrize("selection,message", [
    (select("mmlu", "v1", "x"), "unknown benchmark"),
    (select("tool-probes", "local-tools-v2", "tools/no-call-v1"), "revision"),
    (select("tool-probes", "local-tools-v1", "tools/read-edit-test-v1"), "unknown tool-probes task"),
    (select("niah", "local-niah-v1", "single-middle", "needle-in-a-haystack"), "unknown niah task"),
    (select("tool-episodes", "local-tools-v1", "tools/read-edit-test-v1", split="holdout"), "holdout"),
])
def test_unknown_id_revision_task_and_pretend_holdout_are_rejected(selection, message):
    with pytest.raises(RegistryError, match=message):
        builtin_registry().validate([selection], broker_available=False)


def test_public_suites_without_an_adapter_stay_import_only_and_never_executable():
    registry = builtin_registry()
    wired = {"bfcl", "ruler", "evalplus", "aider-polyglot"}
    import_only = {entry.benchmark_id for entry in registry.entries if entry.capability == "import-only"}
    assert import_only == set(SUITES) - wired
    for name in import_only:
        entry = registry.get(name)
        assert entry.official is True and entry.task_ids == ()
        for broker in (False, True):
            with pytest.raises(RegistryError, match="import-only"):
                registry.validate([select(name, entry.revision, "HumanEval/0")], broker_available=broker)


def test_a_wired_public_suite_is_refused_until_its_dataset_is_baked():
    """The guarantee that replaces import-only for the four adapters: code is not enough, the pinned
    data must be present in this image, and preflight says so before a model is ever loaded."""
    registry = builtin_registry()
    for name in ("bfcl", "ruler", "evalplus", "aider-polyglot"):
        entry = registry.get(name)
        selection = select(name, entry.revision, f"{name}/whatever")
        with pytest.raises(RegistryError, match="requires"):
            registry.validate([selection], broker_available=True)
        registry.validate([selection], broker_available=True, datasets_available=[name])


def test_a_dataset_backed_selection_must_stay_inside_its_namespace():
    registry = builtin_registry()
    entry = registry.get("ruler")
    for task in ("niah/single-middle", "ruler", "ruler/../escape/8192"):
        with pytest.raises(RegistryError, match="unknown ruler task"):
            registry.validate([select("ruler", entry.revision, task)], broker_available=False,
                              datasets_available=["ruler"])


def test_declared_options_are_accepted_and_anything_else_is_refused():
    registry = builtin_registry()
    entry = registry.get("bfcl")
    allowed = select("bfcl", entry.revision, "bfcl/native/simple_0")
    object.__setattr__(allowed, "options", {"modes": ("native",)})
    registry.validate([allowed], broker_available=False, datasets_available=["bfcl"])
    refused = select("bfcl", entry.revision, "bfcl/native/simple_0")
    object.__setattr__(refused, "options", {"shell": "rm -rf /"})
    with pytest.raises(RegistryError, match="does not accept option"):
        registry.validate([refused], broker_available=False, datasets_available=["bfcl"])


def test_coding_needs_the_broker():
    registry = builtin_registry()
    entry = registry.get("coding")
    assert entry.capability == "unavailable" and entry.requires == ("coding-broker",)
    assert set(entry.task_ids) == {item.fixture_id for item in fixtures()}
    assert set(entry.languages) == {"python", "typescript", "javascript"}
    # The registry sees a plain selection object; ContainerRunConfig separately rejects coding in stage 1.
    selection = select("coding", "private-coding-v1", "python/chunks")
    with pytest.raises(RegistryError, match="coding-broker"):
        registry.validate([selection], broker_available=False)
    registry.validate([selection], broker_available=True)
    with pytest.raises(RegistryError):
        registry.validate([selection], broker_available="yes")


def test_coding_is_runnable_only_with_broker_and_never_holdout():
    registry = builtin_registry()
    every = select("coding", "private-coding-v1", *(item.fixture_id for item in fixtures()))
    with pytest.raises(RegistryError, match="coding-broker"):
        registry.validate([every], broker_available=False)
    registry.validate([every], broker_available=True)
    holdout = select("coding", "private-coding-v1", "python/chunks", split="holdout")
    with pytest.raises(RegistryError, match="holdout"):
        registry.validate([holdout], broker_available=True)
    with pytest.raises(RegistryError):
        registry.validate([holdout], broker_available=False)
    with pytest.raises(RegistryError, match="unknown coding task"):
        registry.validate([select("coding", "private-coding-v1", "python/other")], broker_available=True)
    assert registry.get("coding").capability == "unavailable"  # only the wired broker makes it executable


def test_duplicates_empty_selection_and_entry_invariants():
    registry = builtin_registry()
    twice = [select("niah", "local-niah-v1", "multi"), select("niah", "local-niah-v1", "multi", "missing")]
    with pytest.raises(RegistryError, match="twice"):
        registry.validate(twice, broker_available=False)
    registry.validate([twice[0], select("niah", "local-niah-v1", "multi", seed=43)], broker_available=False)
    with pytest.raises(RegistryError):
        registry.validate([], broker_available=False)
    with pytest.raises(ValueError):
        RegistryEntry(benchmark_id="x", revision="v1", category="tools", capability="runner", task_ids=("a",),
                      requires=("coding-broker",))
    with pytest.raises(ValueError):
        RegistryEntry(benchmark_id="x", revision="v1", category="tools", capability="runner")
    with pytest.raises(ValueError):
        RegistryEntry(benchmark_id="x", revision="v1", category="tools", capability="runner", task_ids=("a",),
                      note="unknown fields are rejected")
