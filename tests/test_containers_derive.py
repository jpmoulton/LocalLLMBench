"""Weights -> session config derivation (AM-1 item 1). Pure: synthetic GGUF headers, counted hasher, no model."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmbench.config import canonical_json
from llmbench.containers.config import ContainerRunConfig, read_run_config
from llmbench.containers.derive import (AIDER_LANGUAGES, BFCL_ITEMS_PER_CATEGORY, DEFAULT_CONTEXT_FLOOR,
                                        DEFAULT_CTX_TIERS, DEFAULT_GPU_LAYERS, DEFAULT_KV_OFFLOAD, DEFAULT_KV_PAIRS,
                                        EVALPLUS_ITEM_LIMIT, IMAGE_DATASET_ROOT, MAX_RULER_LENGTHS,
                                        PUBLIC_BENCHMARK_ORDER, RULER_DEFAULT_TASKS, RULER_OUTPUT_TOKENS,
                                        RULER_SKIPPED_TASKS, STAGED_DATASET_DIRNAME, benchmark_plan_for_model,
                                        benchmark_plan_lines, context_allocation_for, context_input_targets,
                                        context_points_budget, context_range_bounds, context_tiers_for_range,
                                        derive_session_config, find_ggufs, public_benchmark_plan, quantization_of,
                                        resolve_dataset_root, ruler_lengths_for, runner_benchmarks,
                                        session_budgets_for, session_id_for)
from llmbench.containers.proposals import apply, baseline_proposal, deterministic_schedule
from llmbench.containers.session import output_reserve_tokens, read_session_config
from llmbench.evaluations.retrieval import NIAH_VARIANTS
from llmbench.registry import builtin_registry
from test_containers_session import EXAMPLE, gguf_bytes


@pytest.fixture(autouse=True)
def away_from_the_staged_datasets(monkeypatch, tmp_path):
    """Derivation's dataset default is resolved against the WORKING DIRECTORY, so these tests run outside the
    repository: whether root has staged real corpora under ``artifacts/benchmark-datasets`` must never decide
    what a derivation test asserts. Tests that want datasets stage their own and pass ``dataset_root``.
    """
    monkeypatch.chdir(tmp_path)


def stage_datasets(root, *, bfcl=True, ruler=True, evalplus=False, aider=False):
    """A real dataset root built from the adapters' own offline fixtures; returns its path as a string."""
    data = Path(__file__).parent / "data"
    root.mkdir(parents=True, exist_ok=True)
    if bfcl:
        from test_benchmarks_bfcl import build_dataset
        shutil.copytree(build_dataset(root / "bfcl-src"), root, dirs_exist_ok=True)
    if ruler:
        from llmbench.benchmarks.ruler_tasks.corpus import ESSAY_JSON
        (root / ESSAY_JSON).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / "ruler-essays.json", root / ESSAY_JSON)
    if evalplus:
        from test_benchmarks_evalplus import write_dataset
        write_dataset(root)
    if aider:
        from llmbench.benchmarks.aider_polyglot import CORPUS_DIRNAME
        shutil.copytree(data / CORPUS_DIRNAME, root / CORPUS_DIRNAME, dirs_exist_ok=True)
    return str(root)


def write_models(directory, quantizations=("Q4_K_M", "Q6_K"), **header):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for quant in quantizations:
        path = directory / f"Qwen3.8-27B-{quant}.gguf"
        path.write_bytes(gguf_bytes(quant, **header))
        paths.append(path)
    return paths


def counting_hasher(calls):
    def hasher(path):
        calls.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hasher


def test_derive_from_directory_hashes_every_gguf_and_builds_the_default_search_space(tmp_path):
    models = write_models(tmp_path / "weights")
    calls = []
    base = read_run_config(EXAMPLE)
    session = derive_session_config(tmp_path / "weights", base=base, hasher=counting_hasher(calls))
    assert sorted(calls) == sorted(models) and len(calls) == 2  # every GGUF hashed exactly once (charged)
    assert [asset.quantization for asset in session.assets] == ["Q4_K_M", "Q6_K"]  # from general.file_type
    assert {asset.model_name for asset in session.assets} == {"Qwen3.8-27B"}
    assert all(asset.sha256 == hashlib.sha256(path.read_bytes()).hexdigest() for asset, path in
               zip(session.assets, models))
    assert all(asset.size_bytes == path.stat().st_size and asset.host_path == str(path.resolve())
               for asset, path in zip(session.assets, models))
    assert session.session_id == "tune-qwen3.8-27b" and session.policy.name == "tune-qwen3.8-27b"
    assert session.search.quantizations == ("Q4_K_M", "Q6_K")
    assert session.search.kv_pairs == DEFAULT_KV_PAIRS == (("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0"))
    assert session.search.ctx_tiers == DEFAULT_CTX_TIERS == (8192, 32768, 131072)  # n_ctx_train 262144 caps nothing
    assert session.search.spec_types == ("none", "draft-mtp")  # the header declares nextn_predict_layers
    assert session.search.reasoning == ("off", "on") and session.search.batch_sizes == (2048,)
    # Every runner-capable LOCAL registry benchmark with all of its task IDs; coding stays out without a broker,
    # and with no staged corpora anywhere (see the autouse fixture) no public suite can be selected either.
    registry = builtin_registry()
    expected = {entry.benchmark_id: set(entry.task_ids) for entry in registry.entries
                if entry.capability == "runner" and entry.task_namespace is None}
    assert {item.benchmark_id: set(item.task_ids) for item in session.base.benchmarks} == expected
    assert "coding" not in expected and {"tool-probes", "tool-episodes", "niah"} <= set(expected)
    assert session.base.dataset_root is None  # nothing staged: the evaluator image's baked path is left alone
    assert all(item.split == "development" and item.seed == 42 for item in session.base.benchmarks)
    assert set(session.holdout.niah_task_ids) == set(NIAH_VARIANTS) and session.holdout.niah_seed == 7
    assert session.budgets.wall_seconds == 14400 and session.budgets.holdout_reserve_seconds == 2400
    assert session.budgets.candidate_wall_seconds == 1800 and session.budgets.cleanup_reserve_seconds == 120
    engine = session.base.engine
    assert (engine.ctx_size, engine.cache_type_k, engine.cache_type_v, engine.flash_attn) == (8192, "f16", "f16", "on")
    assert (engine.batch_size, engine.ubatch_size, engine.spec_type, engine.reasoning) == (2048, 512, "none", "off")
    assert session.base.requested_input_tokens == 4096 and session.base.label == "session-base"
    assert session.base.assets[0] == session.assets[0] and session.image_bundle is None
    assert session.proposal_mode == "deterministic"
    # Reproducible from its own JSON, and identical on a second derivation.
    path = tmp_path / "session-config.json"
    path.write_text(json.dumps(session.model_dump(mode="json")), encoding="utf-8")
    assert read_session_config(path) == session
    assert derive_session_config(tmp_path / "weights", base=base, hasher=counting_hasher([])) == session
    labels = [item.label() for item in deterministic_schedule(session)]
    assert labels[:2] == ["baseline", "kv-q8_0-q8_0"] and "spec-draft-mtp" in labels and "ctx-131072" in labels


def test_derive_from_a_single_file_caps_tiers_by_training_context_and_omits_mtp_without_nextn(tmp_path):
    base = read_run_config(EXAMPLE)
    (path,) = write_models(tmp_path / "one", ("Q4_K_M",), n_ctx_train=32768, nextn=0)
    session = derive_session_config(path, base=base, hasher=counting_hasher([]))
    assert session.search.ctx_tiers == (8192, 32768) and session.search.spec_types == ("none",)
    assert session.search.quantizations == ("Q4_K_M",) and len(session.assets) == 1
    (small,) = write_models(tmp_path / "small", ("Q4_K_M",), n_ctx_train=4096, nextn=0)
    tiny = derive_session_config(small, base=base, hasher=counting_hasher([]))
    assert tiny.search.ctx_tiers == (4096,) and tiny.base.engine.ctx_size == 4096
    assert tiny.base.requested_input_tokens == 2048  # half the only tier keeps output + template reserve in capacity
    (useless,) = write_models(tmp_path / "useless", ("Q4_K_M",), n_ctx_train=256, nextn=0)
    with pytest.raises(ValueError, match="smallest usable tier"):
        derive_session_config(useless, base=base, hasher=counting_hasher([]))
    # The image bundle path and inference image are carried when given; the budget scales the reserves.
    bundled = derive_session_config(path, base=base, hasher=counting_hasher([]), image_bundle="prep/image-bundle.json",
                                    budget_seconds=3600)
    assert bundled.image_bundle == "prep/image-bundle.json"
    assert (bundled.budgets.wall_seconds, bundled.budgets.holdout_reserve_seconds,
            bundled.budgets.candidate_wall_seconds) == (3600, 600, 1800)
    assert session_budgets_for(2400).candidate_wall_seconds == 1800 and session_budgets_for(2400).holdout_reserve_seconds == 400
    # Too short for one candidate above the base stage bounds: refused, never silently shrunk.
    with pytest.raises(ValidationError, match="candidate_wall_seconds must exceed"):
        derive_session_config(path, base=base, hasher=counting_hasher([]), budget_seconds=700)
    with pytest.raises(ValidationError):
        session_budgets_for(500)


def test_derive_rejects_unusable_weights(tmp_path):
    base = read_run_config(EXAMPLE)
    with pytest.raises(ValueError, match="neither a GGUF file nor a directory"):
        derive_session_config(tmp_path / "missing", base=base)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .gguf files"):
        find_ggufs(empty)
    split = tmp_path / "split"
    split.mkdir()
    (split / "model-00001-of-00002.gguf").write_bytes(gguf_bytes("Q4_K_M"))
    with pytest.raises(ValueError, match="split GGUF"):
        find_ggufs(split)
    hidden = tmp_path / "hidden"
    write_models(hidden, ("Q6_K",))
    (hidden / ".partial.gguf").write_bytes(b"x")
    (hidden / "notes.txt").write_bytes(b"x")
    assert [item.name for item in find_ggufs(hidden)] == ["Qwen3.8-27B-Q6_K.gguf"]
    (missing_template,) = write_models(tmp_path / "no-template", ("Q4_K_M",), template="")
    with pytest.raises(ValueError, match="tokenizer.chat_template"):
        derive_session_config(missing_template, base=base, hasher=counting_hasher([]))
    mixed = tmp_path / "mixed"
    write_models(mixed, ("Q4_K_M",))
    (mixed / "Other-Q6_K.gguf").write_bytes(gguf_bytes("Q6_K", name="Other-7B"))
    with pytest.raises(ValueError, match="same model"):
        derive_session_config(mixed, base=base, hasher=counting_hasher([]))
    duplicate = tmp_path / "dup"
    write_models(duplicate, ("Q4_K_M",))
    (duplicate / "copy-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M") + b"\x01")
    with pytest.raises(ValueError, match="same quantization"):
        derive_session_config(duplicate, base=base, hasher=counting_hasher([]))
    with pytest.raises(ValueError, match="holdout seed collides"):
        derive_session_config(tmp_path / "hidden", base=base, hasher=counting_hasher([]), holdout_seed=42)
    # Quantization: header first, filename suffix second, never a guess.
    assert quantization_of(tmp_path / "x-q5_k_m.gguf", {"quantization": None}) == "Q5_K_M"
    assert quantization_of(tmp_path / "x.gguf", {"quantization": "IQ4_XS"}) == "IQ4_XS"
    with pytest.raises(ValueError, match="quantization"):
        quantization_of(tmp_path / "model.gguf", {"quantization": None})
    assert session_id_for("Qwen3.8 27B (Instruct)") == "tune-qwen3.8-27b-instruct" and session_id_for("!!!") == "tune-model"


def test_context_range_replaces_the_default_tiers_with_allocations_for_input_targets(tmp_path):
    """U16: the range names USABLE INPUT tokens; each tier allocates input + template reserve + output."""
    base = read_run_config(EXAMPLE)
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))  # n_ctx_train 262144
    session = derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=131072,
                                    context_ceiling=200000)
    reserve, output = 256, 512  # base.template_reserve_tokens, max(generation.max_output_tokens, speed.output_tokens)
    targets = context_input_targets(131072, 200000)
    assert targets == [131072, 150784, 173568, 200000]  # floor, two geometric interior targets, ceiling
    assert session.search.ctx_tiers == tuple(context_allocation_for(item, template_reserve_tokens=reserve,
                                                                   output_tokens=output) for item in targets)
    assert session.search.ctx_tiers == (131840, 151552, 174336, 200960)
    assert all(tier % 256 == 0 and tier >= item + reserve + output
               for tier, item in zip(session.search.ctx_tiers, targets))
    assert session.search.ctx_tiers != DEFAULT_CTX_TIERS and max(session.search.ctx_tiers) <= 262144
    assert (session.search.context_floor, session.search.context_ceiling) == (131072, 200000)
    assert session.search.context_n_ctx_train == 262144 and session.search.context_points_dropped == ()
    # The baseline runs at the floor and every context candidate at its tier's input target, output reserved.
    assert session.base.engine.ctx_size == 131840 and session.base.requested_input_tokens == 131072
    fills = {}
    for proposal in deterministic_schedule(session):
        config = apply(session, session.base, proposal) if proposal.family == "baseline" else apply(
            session, apply(session, session.base, baseline_proposal(session)), proposal)
        fills[config.engine.ctx_size] = config.requested_input_tokens
    assert [fills[tier] for tier in session.search.ctx_tiers] == targets  # 200000 exactly, never 200960 - 768
    assert all(fill + reserve + output <= tier for tier, fill in fills.items())
    assert 131072 <= min(fills.values()) and max(fills.values()) == 200000
    # Reproducible from the written session config, so resume searches the same range.
    written = tmp_path / "session-config.json"
    written.write_text(json.dumps(session.model_dump(mode="json")), encoding="utf-8")
    assert read_session_config(written) == session


def test_context_range_defaults_the_missing_bound_and_caps_the_point_count(tmp_path):
    base = read_run_config(EXAMPLE)
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    only_ceiling = derive_session_config(path, base=base, hasher=counting_hasher([]), context_ceiling=32768)
    assert (only_ceiling.search.context_floor, only_ceiling.search.context_ceiling) == (DEFAULT_CONTEXT_FLOOR, 32768)
    only_floor = derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=131072)
    assert only_floor.search.context_ceiling == 261376 == (262144 - 768) // 256 * 256  # the model's largest input
    assert context_range_bounds(None, None, n_ctx_train=262144, template_reserve_tokens=256,
                                output_tokens=512) == (4096, 261376)
    tiny = context_range_bounds(None, 1024, n_ctx_train=8192, template_reserve_tokens=256, output_tokens=512)
    assert tiny == (1024, 1024)  # the default floor never rises above the ceiling
    # The count cap keeps floor and ceiling and records the interior targets it dropped.
    assert context_points_budget(max_candidates=12, quantizations=1, kv_pairs=3, spec_types=2, reasoning=2,
                                 batch_sizes=1) == 4
    assert context_points_budget(max_candidates=12, quantizations=6, kv_pairs=3, spec_types=2, reasoning=2,
                                 batch_sizes=1) == 3
    assert context_points_budget(max_candidates=2, quantizations=4, kv_pairs=3, spec_types=2, reasoning=2,
                                 batch_sizes=1) == 1  # at least the ceiling is always measured
    tiers, dropped = context_tiers_for_range(2048, 32768, template_reserve_tokens=256, output_tokens=512,
                                             n_ctx_train=262144, max_points=2)
    assert tiers == (2816, 33536) and dropped == (5120, 12800)
    assert context_tiers_for_range(2048, 32768, template_reserve_tokens=256, output_tokens=512, n_ctx_train=262144,
                                   max_points=1) == ((33536,), (2048, 5120, 12800))
    six = write_models(tmp_path / "six", ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"))
    capped = derive_session_config(tmp_path / "six", base=base, hasher=counting_hasher([]), context_floor=2048,
                                  context_ceiling=32768)
    assert len(six) == 6 and capped.search.ctx_tiers == (2816, 13568, 33536)
    assert capped.search.context_points_dropped == (5120,)  # a capped search is a subset of the full layout
    assert len(deterministic_schedule(capped)) <= capped.budgets.max_candidates
    labels = [item.label() for item in deterministic_schedule(capped)]
    assert labels[-1] == "ctx-33536" and labels.count("ctx-33536") == 1  # the ceiling still runs


def test_offload_axes_are_opt_in_and_reserve_their_schedule_slots_before_the_context_tiers(tmp_path):
    """U17: the derived default is unchanged, and an asked-for offload sweep never costs the context ceiling."""
    from llmbench.containers.proposals import full_schedule
    base = read_run_config(EXAMPLE)
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    default = derive_session_config(path, base=base, hasher=counting_hasher([]))
    assert default.search.gpu_layers == DEFAULT_GPU_LAYERS == ("all",)
    assert default.search.kv_offload == DEFAULT_KV_OFFLOAD == (True,)
    assert (default.base.engine.n_gpu_layers, default.base.engine.kv_offload) == ("all", True)
    assert [item.label() for item in deterministic_schedule(default)
            if item.family in {"offload", "kv-placement"}] == []  # nothing enters the schedule by default
    # A single-member axis costs no slot, so an existing caller's budget is byte for byte the old answer.
    unchanged = dict(max_candidates=12, quantizations=1, kv_pairs=3, spec_types=2, reasoning=2, batch_sizes=1)
    assert context_points_budget(**unchanged) == context_points_budget(**unchanged, gpu_layers=1,
                                                                       kv_offload=1) == 4
    # Extra members sit BEFORE the context sweep, so they must be counted or the ceiling would be truncated away.
    assert context_points_budget(**unchanged, gpu_layers=5, kv_offload=2) == 3
    assert context_points_budget(**unchanged, gpu_layers=9, kv_offload=4) == 1  # at least the ceiling survives
    assert context_points_budget(max_candidates=2, quantizations=1, kv_pairs=1, spec_types=1, reasoning=1,
                                 batch_sizes=1, gpu_layers=8, kv_offload=2) == 1
    swept = derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=2048,
                                  context_ceiling=32768, gpu_layers=("all", 32, 24, 16, 0),
                                  kv_offload=(True, False))
    assert swept.search.gpu_layers == ("all", 32, 24, 16, 0) and swept.search.kv_offload == (True, False)
    assert swept.base.engine.n_gpu_layers == "all" and swept.base.engine.kv_offload is True  # baseline members
    assert len(swept.search.ctx_tiers) == 3 and swept.search.context_points_dropped  # a target made room
    schedule = deterministic_schedule(swept)
    labels = [item.label() for item in schedule]
    # Nothing is truncated: the full schedule fits the cap and the ceiling tier is still the last candidate.
    assert labels == [item.label() for item in full_schedule(swept)]
    assert len(schedule) <= swept.budgets.max_candidates == 12
    assert labels[-1] == f"ctx-{max(swept.search.ctx_tiers)}"
    assert {"offload-32", "offload-24", "offload-16", "offload-0", "kv-cache-ram"} <= set(labels)
    # Reproducible from the written session config, so resume searches the same offload configurations.
    written = tmp_path / "session-config.json"
    written.write_text(json.dumps(swept.model_dump(mode="json")), encoding="utf-8")
    assert read_session_config(written) == swept


def test_context_range_never_exceeds_the_training_context(tmp_path):
    base = read_run_config(EXAMPLE)
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",), n_ctx_train=8192, nextn=0)
    with pytest.raises(ValueError, match="training context is 8192 tokens"):
        derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=2048, context_ceiling=7800)
    with pytest.raises(ValueError, match="at least 512"):
        derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=128, context_ceiling=4096)
    with pytest.raises(ValueError, match="exceeds context_ceiling"):
        derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=4096, context_ceiling=2048)
    fitting = derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=1024,
                                    context_ceiling=7424)
    assert max(fitting.search.ctx_tiers) == 8192 == context_allocation_for(7424, template_reserve_tokens=256,
                                                                          output_tokens=512)
    assert fitting.search.context_n_ctx_train == 8192
    with pytest.raises(ValueError, match="at least 512 usable input tokens"):
        context_allocation_for(511, template_reserve_tokens=256, output_tokens=512)
    with pytest.raises(ValueError, match="at least one input target"):
        context_input_targets(2048, 4096, max_points=0)


def test_an_auto_derived_ceiling_is_one_this_module_accepts(tmp_path):
    """REV-LIVE-03: the missing ceiling subtracts the ROUNDED reserve, or derivation refuses its own ceiling.

    ``context_allocation_for`` rounds ``input + reserve + output`` UP to 256, so subtracting the raw reserve
    derives a ceiling whose allocation exceeds ``n_ctx_train`` whenever ``n_ctx_train % 256 >= R % 256 > 0``
    (R = reserve + output). The user-facing symptom was ``tune --context-floor N`` failing with a message about
    a ceiling they never typed.
    """
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["generation"]["max_output_tokens"] = raw["speed"]["output_tokens"] = 16  # R = 272: not 256-aligned
    base = ContainerRunConfig.model_validate_json(canonical_json(raw))
    assert (base.template_reserve_tokens, output_reserve_tokens(base)) == (256, 16)
    for n_ctx_train, ceiling in ((40000, 39424), (1000000, 999424)):
        assert context_range_bounds(None, None, n_ctx_train=n_ctx_train, template_reserve_tokens=256,
                                    output_tokens=16) == (DEFAULT_CONTEXT_FLOOR, ceiling)
        allocation = context_allocation_for(ceiling, template_reserve_tokens=256, output_tokens=16)
        assert allocation <= n_ctx_train  # the ceiling this module derives is one it will accept
        (path,) = write_models(tmp_path / f"w{n_ctx_train}", ("Q4_K_M",), n_ctx_train=n_ctx_train)
        session = derive_session_config(path, base=base, hasher=counting_hasher([]), context_floor=4096)
        assert session.search.context_ceiling == ceiling and max(session.search.ctx_tiers) == allocation
        assert session.search.context_n_ctx_train == n_ctx_train
        labels = [item.label() for item in deterministic_schedule(session)]
        assert labels[-1] == f"ctx-{allocation}"  # and the derived schedule still reaches it
    # A no-op wherever the reserve is already a whole multiple of the granularity (every shipped base config).
    assert context_range_bounds(None, None, n_ctx_train=262144, template_reserve_tokens=256,
                                output_tokens=512) == (DEFAULT_CONTEXT_FLOOR, 261376)


def rows_of(plan):
    return {row["benchmark_id"]: row for row in plan["benchmarks"]}


def broker_base():
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["broker"] = {"max_requests": 2}
    return ContainerRunConfig.model_validate_json(canonical_json(raw))


def test_derivation_selects_exactly_the_public_benchmarks_whose_corpora_are_staged(tmp_path):
    """U18: the headline command measures the official suites that are actually there, and nothing else.

    Availability is the evaluator's own probe (`container_eval.baked_datasets`) and every task list comes from the
    adapter, so a selection derivation writes is one the runner will admit rather than refuse at preflight.
    """
    from llmbench.container_eval import baked_datasets
    root = stage_datasets(tmp_path / "datasets")  # BFCL + RULER only; no EvalPlus, no Aider corpus
    assert set(baked_datasets(root)) == {"bfcl", "ruler"}
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    session = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]),
                                    dataset_root=root)
    assert session.base.dataset_root == str(tmp_path / "datasets")  # recorded, so every candidate reads it
    selected = {item.benchmark_id: item for item in session.base.benchmarks
                if item.benchmark_id in PUBLIC_BENCHMARK_ORDER}
    assert set(selected) == {"bfcl", "ruler"}  # exactly the staged ones, never a guess at the others
    assert all(item.split == "development" and item.seed == 42 for item in selected.values())
    # RULER: the two tasks root measured clean, at the only standard length the 8192 baseline tier can hold.
    assert selected["ruler"].task_ids == ("ruler/niah_multikey_3/4096", "ruler/vt/4096")
    assert dict(selected["ruler"].options) == {"tasks": tuple(RULER_DEFAULT_TASKS), "lengths": (4096,),
                                               "ruler_output_tokens": RULER_OUTPUT_TOKENS}
    assert RULER_OUTPUT_TOKENS == 512 and RULER_DEFAULT_TASKS == ("niah_multikey_3", "vt")
    assert set(RULER_SKIPPED_TASKS) == {"cwe", "niah_multiquery"}
    assert not any(task in item for item in selected["ruler"].task_ids for task in RULER_SKIPPED_TASKS)
    # BFCL: native mode only, the first three records per category, exactly what the adapter enumerates.
    from llmbench.container_eval import BENCHMARK_ADAPTERS
    from llmbench.containers.derive import _adapter_context
    expected = BENCHMARK_ADAPTERS["bfcl"]().task_ids(
        _adapter_context("bfcl", root, {"modes": ["native"], "items_per_category": BFCL_ITEMS_PER_CATEGORY}))
    assert selected["bfcl"].task_ids == tuple(expected) and len(expected) > 0
    assert all(item.startswith("bfcl/native/") for item in selected["bfcl"].task_ids)
    assert dict(selected["bfcl"].options) == {"modes": ("native",),
                                             "items_per_category": BFCL_ITEMS_PER_CATEGORY}
    # The local fixtures are still all there: the public suites are added, never substituted.
    assert {"tool-probes", "tool-episodes", "niah"} <= {item.benchmark_id for item in session.base.benchmarks}
    # And the whole selection is one the registry would admit with these datasets and no broker.
    builtin_registry().validate(session.base.benchmarks, broker_available=False,
                                datasets_available=baked_datasets(root))


def test_a_missing_dataset_is_a_recorded_skip_with_a_reason_and_never_a_selection(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    session = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]),
                                    dataset_root=str(empty))
    assert not any(item.benchmark_id in PUBLIC_BENCHMARK_ORDER for item in session.base.benchmarks)
    plan = benchmark_plan_for_model(path, base=read_run_config(EXAMPLE), dataset_root=str(empty))
    assert plan["offered"] == list(PUBLIC_BENCHMARK_ORDER) and plan["selected"] == []
    assert plan["skipped"] == list(PUBLIC_BENCHMARK_ORDER) and plan["datasets_present"] == []
    # Offers are driven off the registry, so a public suite wired in later is offered-and-explained, never absent.
    assert set(plan["offered"]) == {entry.benchmark_id for entry in builtin_registry().entries
                                   if entry.capability == "runner" and entry.task_namespace is not None}
    for benchmark_id, row in rows_of(plan).items():
        assert row["status"] == "skipped" and row["items"] == 0 and row["reason"]
        if benchmark_id in ("bfcl", "ruler"):  # the coding suites report the broker gate they hit first
            assert "not present" in row["reason"] and str(empty) in row["reason"], benchmark_id
    assert any(str(empty) in line for line in benchmark_plan_lines(plan))
    # A root holding only one corpus selects only that one, and says why the other is absent.
    partial = stage_datasets(tmp_path / "ruler-only", bfcl=False)
    plan = benchmark_plan_for_model(path, base=read_run_config(EXAMPLE), dataset_root=partial)
    assert plan["selected"] == ["ruler"] and "bfcl" in plan["skipped"]
    assert "not present" in rows_of(plan)["bfcl"]["reason"]


def test_coding_capable_suites_need_a_broker_and_stay_inside_the_candidate_wall(tmp_path):
    """EvalPlus and Aider Polyglot execute generated code, so the broker rule that gates the private coding
    fixtures gates them too; and a suite whose estimate cannot fit one candidate's wall is skipped, not started.
    """
    root = stage_datasets(tmp_path / "datasets", evalplus=True, aider=True)
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    without = benchmark_plan_for_model(path, base=read_run_config(EXAMPLE), dataset_root=root)
    assert set(without["datasets_present"]) == set(PUBLIC_BENCHMARK_ORDER)
    assert without["selected"] == ["bfcl", "ruler"]  # both coding suites are staged and still not selected
    for benchmark_id in ("evalplus", "aider-polyglot"):
        assert "broker" in rows_of(without)[benchmark_id]["reason"]
    session = derive_session_config(path, base=broker_base(), hasher=counting_hasher([]), dataset_root=root)
    selected = {item.benchmark_id for item in session.base.benchmarks}
    assert "coding" in selected and "evalplus" in selected  # the private fixtures and MBPP+ both need the broker
    evalplus = next(item for item in session.base.benchmarks if item.benchmark_id == "evalplus")
    assert dict(evalplus.options) == {"item_limit": EVALPLUS_ITEM_LIMIT}
    assert len(evalplus.task_ids) == EVALPLUS_ITEM_LIMIT == 20
    assert all(item.startswith("evalplus/mbpp-plus/") for item in evalplus.task_ids)
    aider = next(item for item in session.base.benchmarks if item.benchmark_id == "aider-polyglot")
    assert AIDER_LANGUAGES == ("python",) and dict(aider.options) == {"languages": ("python",)}
    plan = benchmark_plan_for_model(path, base=broker_base(), dataset_root=root)
    assert plan["selected"] == list(PUBLIC_BENCHMARK_ORDER) and plan["wall_planned_seconds"] > 0
    assert plan["wall_planned_seconds"] <= plan["wall_allowance_seconds"]
    # A wall that cannot hold a suite tightens its item count, and only skips when even the smallest will not fit.
    # Both outcomes name the estimate and the allowance, so nothing disappears quietly.
    tightened = public_benchmark_plan(dataset_root=root, dataset_root_source="explicit", broker_configured=True,
                                     usable_input_ceiling=7424, candidate_wall_seconds=500)
    assert tightened["selected"] == ["bfcl", "ruler", "evalplus"]
    assert rows_of(tightened)["evalplus"]["options"] == {"item_limit": 10}
    assert any("tightened" in note for note in rows_of(tightened)["evalplus"]["notes"])
    assert "does not fit" in rows_of(tightened)["aider-polyglot"]["reason"]
    starved = public_benchmark_plan(dataset_root=root, dataset_root_source="explicit", broker_configured=True,
                                   usable_input_ceiling=7424, candidate_wall_seconds=300)
    assert starved["selected"] == ["bfcl", "ruler"]
    for benchmark_id in ("evalplus", "aider-polyglot"):
        reason = rows_of(starved)[benchmark_id]["reason"]
        assert "does not fit" in reason and "allowance" in reason, benchmark_id


def test_ruler_lengths_are_bounded_by_the_smallest_tier_thinned_and_never_truncating(tmp_path):
    """A length that does not fit a candidate's served context scores 0.0, so the ladder is bounded by the
    SMALLEST tier the session will run - and the output cap is raised off upstream's truncating default."""
    from llmbench.benchmarks.ruler import DEFAULT_LENGTHS
    assert ruler_lengths_for(4095) == ((), ())  # nothing fits: RULER is then skipped, not shortened below 4K
    assert ruler_lengths_for(8192) == ((4096, 8192), ())
    kept, dropped = ruler_lengths_for(131072)
    assert kept == (4096, 16384, 32768, 131072) and dropped == (8192, 65536)
    assert len(kept) == MAX_RULER_LENGTHS and set(kept) <= set(DEFAULT_LENGTHS)
    root = stage_datasets(tmp_path / "datasets")
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    tiny = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]),
                                 dataset_root=root, context_floor=2048, context_ceiling=2048)
    assert tiny.search.ctx_tiers == (2816,) and "ruler" not in {row.benchmark_id for row in tiny.base.benchmarks}
    plan = benchmark_plan_for_model(path, base=read_run_config(EXAMPLE), dataset_root=root, context_floor=2048,
                                   context_ceiling=2048)
    assert "2048 usable input tokens" in rows_of(plan)["ruler"]["reason"]
    # At a 131072 floor every candidate holds the whole ladder, and the derived session still validates.
    long = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]),
                                 dataset_root=root, context_floor=131072, context_ceiling=131072)
    ruler = next(row for row in long.base.benchmarks if row.benchmark_id == "ruler")
    assert ruler.options["lengths"] == (4096, 16384, 32768, 131072)
    assert len(ruler.task_ids) == len(RULER_DEFAULT_TASKS) * MAX_RULER_LENGTHS == 8
    assert max(ruler.options["lengths"]) + ruler.options["ruler_output_tokens"] <= long.search.ctx_tiers[0]
    assert long.base.engine.ctx_size == max(long.search.ctx_tiers) and len(long.search.ctx_tiers) == 1
    # Candidate cap and context ceiling guarantees are untouched: a multi-tier search still runs its ceiling
    # last, and its RULER ladder is bounded by the SMALLEST tier so no candidate refuses an item it cannot hold.
    spread = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]),
                                   dataset_root=root, context_floor=32768, context_ceiling=131072)
    schedule = deterministic_schedule(spread)
    assert len(schedule) <= spread.budgets.max_candidates
    assert [item.label() for item in schedule][-1] == f"ctx-{max(spread.search.ctx_tiers)}"
    ladder = next(row for row in spread.base.benchmarks if row.benchmark_id == "ruler").options
    smallest = min(spread.search.ctx_tiers)
    assert max(ladder["lengths"]) + ladder["ruler_output_tokens"] <= smallest
    assert max(ladder["lengths"]) <= smallest - 256 - 512 < max(spread.search.ctx_tiers)
    written = tmp_path / "session-config.json"
    written.write_text(json.dumps(spread.model_dump(mode="json")), encoding="utf-8")
    assert read_session_config(written) == spread  # options and dataset_root survive the round trip


def test_the_dataset_root_default_is_the_staged_directory_only_when_it_exists(tmp_path):
    base = read_run_config(EXAMPLE)
    assert base.evaluator.mode == "host-process" and base.dataset_root is None
    assert resolve_dataset_root(base, project_root=tmp_path) == (None, "none-staged")
    # The working directory is the default project root, which is what `llmbench tune` runs from.
    assert Path.cwd() == tmp_path and resolve_dataset_root(base) == (None, "none-staged")
    staged = tmp_path / STAGED_DATASET_DIRNAME
    staged.mkdir(parents=True)
    assert resolve_dataset_root(base, project_root=tmp_path) == (str(staged.resolve()), "staged-host-default")
    assert resolve_dataset_root(base) == (str(staged.resolve()), "staged-host-default")
    # An explicit root wins, and one that is not a directory is refused rather than guessed.
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert resolve_dataset_root(base, dataset_root=str(other)) == (str(other.resolve()), "explicit")
    with pytest.raises(ValueError, match="is not a directory"):
        resolve_dataset_root(base, dataset_root=str(tmp_path / "absent"))
    # The base config's own value is honoured before any default.
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["dataset_root"] = "/opt/elsewhere"
    named = ContainerRunConfig.model_validate_json(canonical_json(raw))
    assert resolve_dataset_root(named, project_root=tmp_path) == ("/opt/elsewhere", "base-config")
    # A container evaluator's datasets live in the image, which this host cannot enumerate: nothing is claimed.
    raw["dataset_root"] = None
    raw["evaluator"] = {"mode": "container", "image": {"role": "evaluator", "reference": "sha256:" + "b" * 64,
                                                       "image_id": "sha256:" + "b" * 64,
                                                       "entrypoint": ["python", "-m", "llmbench.container_eval"]}}
    contained = ContainerRunConfig.model_validate_json(canonical_json(raw))
    assert resolve_dataset_root(contained, project_root=tmp_path) == (None, "evaluator-image")
    plan = public_benchmark_plan(dataset_root=None, dataset_root_source="evaluator-image", broker_configured=True,
                                usable_input_ceiling=131072, candidate_wall_seconds=1800)
    assert plan["selected"] == [] and plan["datasets_present"] == []
    assert IMAGE_DATASET_ROOT in plan["dataset_root_reason"]
    assert all(IMAGE_DATASET_ROOT in row["reason"] for row in plan["benchmarks"])


def test_the_printed_plan_is_the_plan_the_session_was_derived_from(tmp_path):
    """``llmbench tune`` prints the plan before the run; it must be the same plan, not a second opinion."""
    root = stage_datasets(tmp_path / "datasets", evalplus=True)
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    for base in (read_run_config(EXAMPLE), broker_base()):
        for bounds in ({}, {"context_floor": 131072, "context_ceiling": 131072}):
            session = derive_session_config(path, base=base, hasher=counting_hasher([]), dataset_root=root,
                                            **bounds)
            plan = benchmark_plan_for_model(path, base=base, dataset_root=root, **bounds)
            derived = [{"benchmark_id": item.benchmark_id, "revision": item.revision,
                        "task_ids": list(item.task_ids), "split": item.split, "seed": item.seed,
                        "options": {key: list(value) if isinstance(value, tuple) else value
                                    for key, value in item.options.items()}}
                       for item in session.base.benchmarks if item.benchmark_id in PUBLIC_BENCHMARK_ORDER]
            assert derived == plan["selections"]
            lines = benchmark_plan_lines(plan)
            assert all(any(benchmark_id in line for line in lines) for benchmark_id in PUBLIC_BENCHMARK_ORDER)
            assert all(row["reason"] for row in plan["benchmarks"])  # no silent absence anywhere


def test_coding_benchmark_is_derived_only_when_the_base_carries_a_broker(tmp_path):
    assert [row["benchmark_id"] for row in runner_benchmarks(broker_configured=False)] == ["niah", "tool-episodes",
                                                                                            "tool-probes"]
    with_coding = runner_benchmarks(broker_configured=True)
    coding = next(row for row in with_coding if row["benchmark_id"] == "coding")
    assert coding["revision"] == "private-coding-v1" and coding["task_ids"]
    assert set(coding["task_ids"]) == set(builtin_registry().get("coding").task_ids)
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["broker"] = {"max_requests": 2}  # ContainerRunConfig.broker (Branch A) is what enables coding
    base = ContainerRunConfig.model_validate_json(canonical_json(raw))
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    session = derive_session_config(path, base=base, hasher=counting_hasher([]))
    assert "coding" in {item.benchmark_id for item in session.base.benchmarks} and session.base.broker is not None
    plain = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]))
    assert "coding" not in {item.benchmark_id for item in plain.base.benchmarks}
