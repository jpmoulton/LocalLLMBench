"""Session schema, policy mapping, run-config and manifest mapping. Shared fakes for the campaign tests.

No Docker, model, GPU or network: the fake runner writes the same files a real ContainerRunner would.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmbench.analysis import infer_comparison_family
from llmbench.config import ExperimentManifest, RunMode, canonical_json
from llmbench.containers.config import (ArtifactEntry, BenchmarkSelection, CleanupEvidence, ContainerRunConfig,
                                        ContainerRunResult, ImageBundle, ImageRef)
from llmbench.containers.gguf import FILE_TYPES, template_hash
from llmbench.containers.proposals import Proposal, baseline_proposal
from llmbench.containers.report import summarize_evaluation
from llmbench.containers.session import (ContainerSessionConfig, HoldoutPlan, SearchSpace, SessionBudgets,
                                         asset_metadata, check_context_range_against_headers,
                                         output_reserve_tokens, policy_for, run_config_for, to_manifest, tune,
                                         usable_input_tokens)
from llmbench.measurement import SpeedObservation
from llmbench.safety import SessionLock
from test_containers_gguf import build_gguf

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"
TEMPLATE = "{{ bos_token }}{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
PILOT_TEMPLATE_ID = "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
EPOCH = 1_800_000_000.0
LOCK = SessionLock(True, True, True)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def wall(self):
        return EPOCH + self.now


def gguf_bytes(quantization: str, *, template=TEMPLATE, name="Qwen3.8-27B", n_ctx_train=262144, nextn=1) -> bytes:
    """A small but genuine GGUF v3 header so the real reader (CLI paths) and the fake reader agree."""
    code = {value: key for key, value in FILE_TYPES.items()}[quantization]
    entries = [("general.architecture", "str", "qwen3next"), ("general.name", "str", name),
               ("general.file_type", "u32", code), ("qwen3next.context_length", "u32", n_ctx_train),
               ("qwen3next.block_count", "u32", 48), ("tokenizer.chat_template", "str", template)]
    if nextn:
        entries.append(("qwen3next.nextn_predict_layers", "u32", nextn))
    return build_gguf(entries, tensors=1) + bytes(64)


def model_file(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def asset_for(path: Path, quantization: str, name="Qwen3.8-27B") -> dict:
    return {"role": "main", "host_path": str(path), "container_path": "/models/model.gguf",
            "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "format": "gguf", "quantization": quantization, "model_name": name,
            "source_revision": "local:test-models"}


def make_session(tmp_path, *, quantizations=("Q4_K_M", "Q6_K"), kv_pairs=(("f16", "f16"), ("q8_0", "q8_0")),
                 ctx_tiers=(8192, 32768), spec_types=("none", "draft-mtp"), reasoning=("off", "on"),
                 batch_sizes=(2048,), budgets=None, holdout_seed=7, session_id="session-test",
                 model_name="Qwen3.8-27B", context_floor=None, context_ceiling=None, context_n_ctx_train=None,
                 context_points_dropped=(), gpu_layers=None, kv_offload=None, **overrides) -> ContainerSessionConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    assets = [asset_for(model_file(tmp_path, f"model-{quant}.gguf", gguf_bytes(quant, name=model_name)), quant,
                        model_name) for quant in quantizations]
    base = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    base["assets"] = [assets[0]]
    base["label"] = "base"
    context = {} if context_floor is None and context_ceiling is None else {
        "context_floor": context_floor, "context_ceiling": context_ceiling,
        "context_n_ctx_train": context_n_ctx_train, "context_points_dropped": list(context_points_dropped)}
    # U17: the offload axes are left OUT of the search dict unless a test asks for them, so the default fixture
    # is exactly the session shape that existed before the axes did.
    offload = {name: list(values) for name, values in (("gpu_layers", gpu_layers), ("kv_offload", kv_offload))
               if values is not None}
    # REV-LIVE-04: a declared ceiling must survive the candidate cap, so a ranged fixture gets room for the whole
    # context sweep; a test that exercises the cap itself passes its own budgets.
    default_budgets = {"wall_seconds": 3600, "holdout_reserve_seconds": 900, "cleanup_reserve_seconds": 60,
                       "max_candidates": 12 if context else 4, "candidate_wall_seconds": 900,
                       "max_validation_candidates": 2}
    raw = {"schema_version": 1, "session_id": session_id, "assets": assets, "base": base,
           "search": {"quantizations": list(quantizations), "kv_pairs": [list(pair) for pair in kv_pairs],
                      "ctx_tiers": list(ctx_tiers), "spec_types": list(spec_types), "reasoning": list(reasoning),
                      "batch_sizes": list(batch_sizes), **offload, **context},
           "holdout": {"niah_task_ids": ["single-middle", "multi"], "niah_seed": holdout_seed},
           "budgets": budgets or default_budgets,
           **overrides}
    return ContainerSessionConfig.model_validate_json(canonical_json(raw))


def make_bundle(reference_suffix="a" * 64) -> ImageBundle:
    inference = json.loads(EXAMPLE.read_text(encoding="utf-8"))["inference_image"]
    inference.update(reference="ghcr.io/ggml-org/llama.cpp@sha256:" + reference_suffix,
                     image_id="sha256:" + reference_suffix)
    return ImageBundle(prepared_utc="2026-09-18T00:00:00+00:00", inference=ImageRef.model_validate_json(
        json.dumps(inference)), evaluator=ImageRef(role="evaluator", reference="sha256:" + "b" * 64,
                                                  image_id="sha256:" + "b" * 64,
                                                  entrypoint=("python", "-m", "llmbench.container_eval")),
                        help_sha256=inference["help_sha256"], registry_digest="c" * 64)


def observation(config, rate=69.0):
    duration = 512 / rate
    return SpeedObservation(status="completed", finish_reason="length", output_tokens=512,
                            native_generation_seconds=duration, elapsed_seconds=duration + 0.6,
                            first_event_seconds=0.4, content_event_times=tuple(0.4 + duration * i / 99 for i in range(100)),
                            requested_output_tokens=512, input_tokens=config.requested_input_tokens,
                            expected_input_tokens=config.requested_input_tokens, accepted_tokens_verified=True,
                            prompt_processing_seconds=1.2, native_timing_source="llama.cpp.timings")


def fake_evaluation(config: ContainerRunConfig, template: str, *, rate=69.0, retrieval_score=1.0) -> dict:
    samples = []
    for selection in config.benchmarks:
        for task in selection.task_ids:
            if selection.benchmark_id == "niah":
                task_id = f"niah/{task}/{selection.split}/seed-{selection.seed}"
                count = config.requested_input_tokens - 5
                samples.append({"task_id": task_id, "category": "retrieval", "suite": "niah", "split": selection.split,
                                "status": "completed", "score": retrieval_score, "passed": retrieval_score == 1.0,
                                "fixture_hash": hashlib.sha256(task_id.encode()).hexdigest(), "synthetic": False,
                                "model_evaluated": True, "outcome_status": "passed",
                                "context": {"actual_input_tokens": count, "observed_input_tokens": count,
                                            "actual_context_verified": True, "counting_method": "exact-post-template",
                                            "truncation_reported": False, "context_policy": "reject-overflow-no-shift",
                                            "tokenizer_id": "gguf-sha256:" + config.model.sha256,
                                            "template_id": template_hash(template), "requested_input_tokens":
                                            config.requested_input_tokens, "context_capacity": config.engine.ctx_size}})
            else:
                samples.append({"task_id": task, "category": "tools", "suite": selection.benchmark_id,
                                "split": selection.split, "status": "completed", "score": 1.0, "passed": True,
                                "fixture_hash": hashlib.sha256(task.encode()).hexdigest(), "synthetic": False,
                                "model_evaluated": True, "outcome_status": "passed"})
    observations = [json.loads(json.dumps(asdict(observation(config, rate)))) for _ in range(config.speed.repetitions)]
    return {"samples": samples, "speed_observations": observations,
            "readback": {"props": {"chat_template": template, "model_alias": config.alias()}, "models": {}, "slots": []},
            "count_route": {"exact": True}, "overflow_probe": {"passed": True}, "actual_context_verified": True,
            "abort_reason": None, "errors": [], "warmups": [], "native_timings": [], "stages": []}


class FakeRunner:
    """Writes the files a real ContainerRunner leaves behind, scripted per candidate label."""

    def __init__(self, clock: FakeClock, *, template=TEMPLATE, script=None, policy=None):
        self.clock, self.template, self.script, self.policy = clock, template, script or {}, policy
        self.calls = []

    def run(self, config, output_dir, *, remaining_budget_seconds=None, lease_held=False):
        assert lease_held is True, "the campaign lock already holds the shared GPU lease path"
        run_dir = Path(output_dir)
        assert not run_dir.exists()
        self.calls.append({"label": config.label, "run_dir": run_dir, "budget": remaining_budget_seconds,
                           "config": config})
        step = self.script.get(config.label, {})
        if "raise" in step:
            raise step["raise"]
        run_dir.mkdir(parents=True)
        state = step.get("state", "completed")
        evaluation = fake_evaluation(config, step.get("template", self.template), rate=step.get("rate", 69.0),
                                     retrieval_score=step.get("retrieval_score", 1.0))
        files = {"config.json": (canonical_json(config.model_dump(mode="json")) + "\n").encode(),
                 "settings-evidence.json": b'{"settings": [], "unverified_required": []}\n',
                 "stages.jsonl": b'{"name": "admit", "status": "ok"}\n',
                 "reports/candidate.md": f"# Container candidate `{config.label}`\n".encode()}
        if state != "rejected":
            files["evaluation.json"] = json.dumps(evaluation, indent=1).encode()
        for name, content in files.items():
            target = run_dir / Path(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        entries = [{"path": name, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
                   for name, content in files.items()]
        summary = summarize_evaluation(config, evaluation if state != "rejected" else {}, self.policy)
        self.clock.now += step.get("seconds", 120.0)
        result = ContainerRunResult(
            session_id=config.session_id or "none", attempt_id=hashlib.md5(config.label.encode()).hexdigest(),
            config_fingerprint=config.fingerprint(), project_name="llmbench-test", state=state, synthetic=False,
            failure_stage=step.get("stage", None if state == "completed" else "quality"),
            failure_reasons=tuple(step.get("reasons", () if state == "completed" else (f"scripted {state}",))),
            warnings=tuple(step.get("warnings", ())), started_utc="2026-09-18T00:00:00+00:00",
            finished_utc="2026-09-18T00:02:00+00:00", elapsed_seconds=step.get("seconds", 120.0),
            budget_charged_seconds=step.get("seconds", 120.0), effective_settings_verified=state == "completed",
            actual_context_verified=state == "completed" and evaluation["actual_context_verified"],
            speed=summary["speed"], quality=summary["quality"], samples_total=summary["samples_total"],
            load_seconds=63.6, vram_used_mib_after_load=21179.0,
            cleanup=CleanupEvidence(attempted=True, verified=state != "cleanup-uncertain",
                                    error=None if state != "cleanup-uncertain" else "owned container remains"),
            abort_campaign=state == "cleanup-uncertain",
            artifacts=tuple(ArtifactEntry(**entry) for entry in entries),
            reports={"candidate": "reports/candidate.md", "markdown": "reports/report.md"})
        payload = (json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n").encode()
        (run_dir / "result.json").write_bytes(payload)
        entries.append({"path": "result.json", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
        (run_dir / "artifact-index.json").write_text(json.dumps({"schema_version": 1, "artifacts": sorted(
            entries, key=lambda e: e["path"])}), encoding="utf-8")
        if step.get("corrupt_index"):
            (run_dir / "evaluation.json").write_bytes(b"{}")
        return result


def metadata_reader(template=TEMPLATE):
    def read(path):
        return {"template_hash": template_hash(template), "block_count": 48, "architecture": "qwen3next",
                "name": "Qwen3.8-27B", "n_ctx_train": 262144, "nextn_predict_layers": 1}
    return read


def development_labels(runner) -> list[str]:
    """The development schedule a runner was asked to execute, without the holdout attempts that follow it.

    Finalists are re-run on held-out items once the development candidates finish, so a complete session makes
    `<label>-holdout` calls too. Tests about the development schedule read it through this.
    """
    return [call["label"] for call in runner.calls if not call["label"].endswith("-holdout")]


@pytest.fixture
def isolated_gpu_lock(tmp_path, monkeypatch):
    """The controller's live resource lock lives in the temp dir; keep the tests away from a real GPU lease."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(lock_dir))
    return lock_dir


def test_session_config_is_strict_and_axes_validate_through_engine_settings(tmp_path):
    session = make_session(tmp_path)
    assert session.search.kv_pairs == (("f16", "f16"), ("q8_0", "q8_0"))
    assert session.budgets.candidate_wall_seconds == 900 and session.policy.mode == RunMode.LIVE
    raw = json.loads(session.model_dump_json())
    raw["unexpected"] = 1
    with pytest.raises(ValidationError):
        ContainerSessionConfig.model_validate_json(json.dumps(raw))
    with pytest.raises(ValidationError, match="flash_attn on"):  # the engine's K/V kernel rule, no exceptions
        SearchSpace(quantizations=("Q4_K_M",), kv_pairs=(("q8_0", "f16"),), ctx_tiers=(8192,), spec_types=("none",),
                    reasoning=("off",), batch_sizes=(2048,))
    with pytest.raises(ValidationError, match="ascend"):
        SearchSpace(quantizations=("Q4_K_M",), kv_pairs=(("f16", "f16"),), ctx_tiers=(32768, 8192),
                    spec_types=("none",), reasoning=("off",), batch_sizes=(2048,))
    with pytest.raises(ValidationError):
        SearchSpace(quantizations=("Q4_K_M",), kv_pairs=(("f16", "f16"),), ctx_tiers=(8192,), spec_types=("lookahead",),
                    reasoning=("off",), batch_sizes=(2048,))
    with pytest.raises(ValidationError, match="search axis batch_size=16384"):  # batch above the first tier
        make_session(tmp_path / "b", batch_sizes=(2048, 16384))
    with pytest.raises(ValidationError, match="without an asset"):
        make_session(tmp_path / "c", quantizations=("Q4_K_M",), search={"quantizations": ["Q8_0"],
                     "kv_pairs": [["f16", "f16"]], "ctx_tiers": [8192], "spec_types": ["none"], "reasoning": ["off"],
                     "batch_sizes": [2048]})
    with pytest.raises(ValidationError, match="candidate_wall_seconds must exceed"):
        make_session(tmp_path / "d", budgets={"wall_seconds": 3600, "holdout_reserve_seconds": 600,
                                              "candidate_wall_seconds": 600})
    with pytest.raises(ValidationError):
        SessionBudgets(wall_seconds=3600, holdout_reserve_seconds=3000, cleanup_reserve_seconds=600)
    with pytest.raises(ValidationError, match="unknown"):
        HoldoutPlan(niah_task_ids=("single-middle", "nope"), niah_seed=7)
    with pytest.raises(ValidationError, match="proposal_file"):
        make_session(tmp_path / "e", proposal_mode="file")
    with pytest.raises(ValidationError, match="live"):
        make_session(tmp_path / "f", policy={"schema_version": 1, "name": "x", "mode": "offline"})


def _space(**overrides) -> SearchSpace:
    space = {"quantizations": ("Q4_K_M",), "kv_pairs": (("f16", "f16"),), "ctx_tiers": (8192,),
             "spec_types": ("none",), "reasoning": ("off",), "batch_sizes": (2048,)}
    return SearchSpace(**{**space, **overrides})


def test_offload_axes_default_to_todays_behaviour_and_validate_through_engine_settings(tmp_path):
    """U17: CPU/RAM offload is a searchable axis; its default single member is what every session ran before."""
    plain = make_session(tmp_path)  # the pre-U17 fixture shape: the search dict names neither axis
    assert plain.search.gpu_layers == ("all",) and plain.search.kv_offload == (True,)
    assert (plain.base.engine.n_gpu_layers, plain.base.engine.kv_offload) == ("all", True)
    swept = make_session(tmp_path / "s", gpu_layers=("all", 24, 0), kv_offload=(True, False),
                         budgets={"wall_seconds": 14400, "max_candidates": 12})
    assert swept.search.gpu_layers == ("all", 24, 0) and swept.search.kv_offload == (True, False)
    # 0 GPU layers and a RAM KV cache are members, not emptiness: the truthiness rule of the other axes must
    # never be applied to these two.
    assert _space(gpu_layers=(0,), kv_offload=(False,)).gpu_layers == (0,)
    assert _space(gpu_layers=(0,), kv_offload=(False,)).kv_offload == (False,)
    for match, space in (("at least 1 item", {"gpu_layers": ()}), ("at least 1 item", {"kv_offload": ()}),
                         ("gpu_layers must be unique", {"gpu_layers": ("all", "all")}),
                         ("gpu_layers must be unique", {"gpu_layers": (24, 24)}),
                         ("kv_offload must be unique", {"kv_offload": (True, True)}),
                         ("999", {"gpu_layers": (1000,)}), ("greater than or equal", {"gpu_layers": (-1,)}),
                         ("all", {"gpu_layers": ("most",)}), ("bool", {"kv_offload": ("ram",)}),
                         ("bool", {"kv_offload": (1,)})):
        with pytest.raises(ValidationError, match=match):
            _space(**space)


def test_a_base_candidate_whose_offload_settings_are_undeclared_is_refused(tmp_path):
    """U17: the baseline pins every axis, so a silently re-tuned base candidate fails closed instead."""
    raw = json.loads(make_session(tmp_path).model_dump_json())
    assert raw["search"]["gpu_layers"] == ["all"] and raw["search"]["kv_offload"] is not None
    raw["base"]["engine"]["n_gpu_layers"] = 24  # partial offload asked for in the base, not listed in the search
    with pytest.raises(ValidationError, match="does not list"):
        ContainerSessionConfig.model_validate_json(json.dumps(raw))
    raw["search"]["gpu_layers"] = [24, "all"]  # declaring it makes the session coherent and keeps the base's ask
    declared = ContainerSessionConfig.model_validate_json(json.dumps(raw))
    assert declared.base.engine.n_gpu_layers == 24
    raw["search"]["gpu_layers"] = ["all"]
    raw["base"]["engine"]["n_gpu_layers"] = "all"
    raw["base"]["engine"]["kv_offload"] = False  # the same rule for the KV cache placement
    with pytest.raises(ValidationError, match="does not list"):
        ContainerSessionConfig.model_validate_json(json.dumps(raw))


def test_the_context_ceiling_still_survives_the_cap_with_the_offload_families_present(tmp_path):
    """U17 must not disturb REV-LIVE-04: context stays last, so a cap takes the ceiling tier away last."""
    from llmbench.containers.proposals import SWEEP_ORDER, deterministic_schedule, full_schedule
    assert SWEEP_ORDER[-1] == "context" and {"offload", "kv-placement"} <= set(SWEEP_ORDER)
    assert max(SWEEP_ORDER.index("offload"), SWEEP_ORDER.index("kv-placement")) < SWEEP_ORDER.index("context")
    budgets = {"wall_seconds": 5400, "holdout_reserve_seconds": 900, "cleanup_reserve_seconds": 120,
               "max_candidates": 4, "candidate_wall_seconds": 900, "max_validation_candidates": 2}
    shape = dict(quantizations=("Q4_K_M",), spec_types=("none",), reasoning=("off",), kv_pairs=(("f16", "f16"),),
                 ctx_tiers=(4864, 8960), context_floor=4096, context_ceiling=8192, context_n_ctx_train=262144,
                 gpu_layers=("all", 24), kv_offload=(True, False))
    session = make_session(tmp_path, budgets=budgets, **shape)
    labels = [item.label() for item in deterministic_schedule(session)]
    assert labels == [item.label() for item in full_schedule(session)]
    assert labels == ["baseline", "offload-24", "kv-cache-ram", "ctx-8960"]  # offload sweeps, ceiling still runs
    top = run_config_for(session, None, Proposal(family="context", changes={"ctx_tier": 8960}),
                         baseline=baseline_proposal(session))
    assert (top.engine.ctx_size, top.requested_input_tokens) == (8960, 8192)
    # One candidate slot fewer and the offload candidates would push the ceiling out: refused, naming it.
    with pytest.raises(ValidationError) as refusal:
        make_session(tmp_path / "tight", budgets={**budgets, "max_candidates": 3}, **shape)
    assert "context_ceiling 8192 usable input tokens would never be measured" in str(refusal.value)
    assert "raise max_candidates to at least 4" in str(refusal.value)


def test_to_manifest_expresses_partial_and_full_cpu_offload(tmp_path):
    """U17: the offload fraction a candidate really asked for reaches the manifest, 0 GPU layers included."""
    session = make_session(tmp_path, gpu_layers=("all", 24, 0), kv_offload=(True, False),
                           budgets={"wall_seconds": 14400, "max_candidates": 12})
    baseline = baseline_proposal(session)
    partial = run_config_for(session, None, Proposal(family="offload", changes={"gpu_layers": 24}),
                             baseline=baseline)
    assert partial.engine.n_gpu_layers == 24 and partial.label == "offload-24"
    assert to_manifest(partial, template_hash="sha256:" + "0" * 64, scorer_revision="s", environment_hash="e",
                       block_count=48).backend.gpu_offload == 0.5
    none_on_gpu = run_config_for(session, None, Proposal(family="offload", changes={"gpu_layers": 0}),
                                 baseline=baseline)
    # Nothing on the GPU is a known fraction even without the block count; it must not fail closed as a guess.
    assert _manifest(none_on_gpu).backend.gpu_offload == 0.0
    ram_kv = run_config_for(session, None, Proposal(family="kv-placement", changes={"kv_offload": False}),
                            baseline=baseline)
    assert ram_kv.engine.kv_offload is False and ram_kv.label == "kv-cache-ram"
    assert _manifest(ram_kv).backend.kv_on_gpu is False
    # Both controls are mapped engine fields, so they change the backend and not the opaque runtime_revision.
    reference = _manifest(run_config_for(session, None, baseline))
    assert _manifest(ram_kv).backend.runtime_revision == reference.backend.runtime_revision
    assert infer_comparison_family(reference, _manifest(ram_kv)) == "performance"
    assert infer_comparison_family(reference, _manifest(none_on_gpu)) == "performance"


def test_context_range_bounds_usable_input_tokens_with_output_reserved_on_top(tmp_path):
    """U16: context_floor/context_ceiling are USABLE INPUT tokens; the tiers are allocations above them."""
    session = make_session(tmp_path, ctx_tiers=(2816, 6912), context_floor=2048, context_ceiling=6000,
                           context_n_ctx_train=262144)
    base = session.base
    assert (base.template_reserve_tokens, output_reserve_tokens(base)) == (256, 512)  # 768 reserved above input
    assert usable_input_tokens(2816, template_reserve_tokens=256, output_tokens=512) == 2048
    assert usable_input_tokens(6912, template_reserve_tokens=256, output_tokens=512, ceiling=6000) == 6000
    assert session.search.context_range() == {"context_floor": 2048, "context_ceiling": 6000,
                                              "n_ctx_train": 262144, "unit": "usable input tokens",
                                              "dropped_input_targets": []}
    assert make_session(tmp_path / "n").search.context_range() is None  # no range: nothing claimed
    # Every candidate the range produces really fills its tier's input target, output still reserved.
    baseline = run_config_for(session, None, baseline_proposal(session))
    assert (baseline.engine.ctx_size, baseline.requested_input_tokens) == (2816, 2048)
    top = run_config_for(session, None, Proposal(family="context", changes={"ctx_tier": 6912}),
                         baseline=baseline_proposal(session))
    assert (top.engine.ctx_size, top.requested_input_tokens) == (6912, 6000)  # exactly the ceiling, not the tier
    assert top.requested_input_tokens + top.template_reserve_tokens + 512 <= top.engine.ctx_size


def test_unreachable_context_ceiling_is_refused_naming_the_training_context(tmp_path):
    """Fail closed: a ceiling whose allocation does not fit the model's training context is never searched."""
    with pytest.raises(ValidationError, match="training context is 8192 tokens"):
        make_session(tmp_path, ctx_tiers=(8192,), context_floor=2048, context_ceiling=7800,
                     context_n_ctx_train=8192)  # 7800 + 256 reserve + 512 output = 8568 > 8192
    with pytest.raises(ValidationError, match="unreachable"):  # not even the bare input fits
        make_session(tmp_path / "b", ctx_tiers=(8192,), context_floor=2048, context_ceiling=8192,
                     context_n_ctx_train=8192)
    with pytest.raises(ValidationError, match="exceeds the model's training context"):
        make_session(tmp_path / "c", ctx_tiers=(2816, 6912), context_floor=2048, context_ceiling=6000,
                     context_n_ctx_train=6000 + 768)  # a tier above n_ctx_train is never allocated
    # The ceiling must actually be measured by some tier, or "supports N tokens" could never be evidenced.
    with pytest.raises(ValidationError, match="would never be measured"):
        make_session(tmp_path / "d", ctx_tiers=(2816,), context_floor=2048, context_ceiling=6000,
                     context_n_ctx_train=262144)
    with pytest.raises(ValidationError, match="below context_floor"):
        make_session(tmp_path / "e", ctx_tiers=(1024, 6912), context_floor=2048, context_ceiling=6000,
                     context_n_ctx_train=262144)


def test_context_range_is_given_whole_and_checked_against_the_weights(tmp_path):
    for floor, ceiling, expected in ((2048, None, "must be given together"), (None, 6000, "must be given together"),
                                     (6000, 2048, "exceeds context_ceiling"), (256, 6000, "greater than or equal")):
        with pytest.raises(ValidationError, match=expected):
            make_session(tmp_path, ctx_tiers=(2816, 6912), context_floor=floor, context_ceiling=ceiling,
                         context_n_ctx_train=262144)
    with pytest.raises(ValidationError, match="context_n_ctx_train"):  # no training context: nothing to check
        make_session(tmp_path / "b", ctx_tiers=(2816, 6912), context_floor=2048, context_ceiling=6000)
    with pytest.raises(ValidationError, match="belong to a context range"):
        make_session(tmp_path / "c", search={"quantizations": ["Q4_K_M"], "kv_pairs": [["f16", "f16"]],
                                             "ctx_tiers": [8192], "spec_types": ["none"], "reasoning": ["off"],
                                             "batch_sizes": [2048], "context_n_ctx_train": 262144},
                     quantizations=("Q4_K_M",))
    with pytest.raises(ValidationError, match="inside the range"):
        make_session(tmp_path / "d", ctx_tiers=(2816, 6912), context_floor=2048, context_ceiling=6000,
                     context_n_ctx_train=262144, context_points_dropped=(9000,))
    # The GGUF headers get the last word at tune: a declared training context the weights lack is refused.
    session = make_session(tmp_path / "e", ctx_tiers=(2816, 6912), context_floor=2048, context_ceiling=6000,
                           context_n_ctx_train=262144)
    facts = asset_metadata(session, reader=metadata_reader())
    check_context_range_against_headers(session, facts)  # headers declare 262144: no complaint
    smaller = {sha: {**meta, "n_ctx_train": 32768} for sha, meta in facts.items()}
    with pytest.raises(ValueError, match="the weights do not have"):
        check_context_range_against_headers(session, smaller)
    missing = {sha: {**meta, "n_ctx_train": None} for sha, meta in facts.items()}
    with pytest.raises(ValueError, match="declares no training context length"):
        check_context_range_against_headers(session, missing)
    plain = make_session(tmp_path / "f")  # no range: the headers are not consulted for one
    check_context_range_against_headers(plain, {})


def test_a_context_ceiling_the_candidate_cap_would_truncate_away_is_refused(tmp_path):
    """REV-LIVE-04: the ceiling must reach the SCHEDULE, not merely be allocated by the largest tier.

    Exactly the shape the first live campaign shipped (four tiers, two K/V pairs, max_candidates 4): the context
    sweep is last and ascending, so deterministic_schedule truncates the ceiling tier away first and the session
    would declare 4096..8192 usable input tokens while never running a candidate above 6400.
    """
    from llmbench.containers.proposals import deterministic_schedule, full_schedule
    budgets = {"wall_seconds": 5400, "holdout_reserve_seconds": 900, "cleanup_reserve_seconds": 120,
               "max_candidates": 4, "candidate_wall_seconds": 900, "max_validation_candidates": 2}
    shape = dict(quantizations=("Q4_K_M",), spec_types=("none",), reasoning=("off",),
                 ctx_tiers=(4864, 5888, 7168, 8960), context_floor=4096, context_ceiling=8192,
                 context_n_ctx_train=262144)
    with pytest.raises(ValidationError) as refusal:
        make_session(tmp_path, budgets=budgets, **shape)
    message = str(refusal.value)
    assert "context_ceiling 8192 usable input tokens would never be measured" in message
    assert "max_candidates 4" in message and "8960-token ceiling tier" in message
    assert "4 context tiers declared" in message and "raise max_candidates to at least 5" in message
    # One more candidate slot makes the ceiling reachable, and the ceiling really is filled to the declared bound.
    session = make_session(tmp_path / "ok", budgets={**budgets, "max_candidates": 5}, **shape)
    labels = [item.label() for item in deterministic_schedule(session)]
    assert labels == [item.label() for item in full_schedule(session)]
    assert labels == ["baseline", "kv-q8_0-q8_0", "ctx-5888", "ctx-7168", "ctx-8960"]
    top = run_config_for(session, None, Proposal(family="context", changes={"ctx_tier": 8960}),
                         baseline=baseline_proposal(session))
    assert (top.engine.ctx_size, top.requested_input_tokens) == (8960, 8192)
    # A cap that leaves only the baseline is coherent when the baseline IS the ceiling tier.
    single = make_session(tmp_path / "one", budgets={**budgets, "max_candidates": 1}, quantizations=("Q4_K_M",),
                          spec_types=("none",), reasoning=("off",), ctx_tiers=(8960,), context_floor=4096,
                          context_ceiling=8192, context_n_ctx_train=262144)
    assert [item.label() for item in deterministic_schedule(single)] == ["baseline"]
    # A session without a range is untouched by the rule: the cap may cut its context sweep as before.
    capped = make_session(tmp_path / "plain", budgets=budgets, ctx_tiers=(8192, 32768))
    assert len(deterministic_schedule(capped)) == 4 < len(full_schedule(capped))


def test_live_smoke_example_schedules_the_context_ceiling_it_declares():
    """REV-LIVE-04: the shipped live-smoke session dropped its own ceiling; it must now run it."""
    from llmbench.containers.proposals import deterministic_schedule, full_schedule
    from llmbench.containers.session import read_session_config
    session = read_session_config(EXAMPLE.with_name("session-smoke.json"))
    assert (session.search.context_floor, session.search.context_ceiling) == (4096, 8192)
    assert session.search.ctx_tiers == (4864, 5888, 7168, 8960) and session.search.context_n_ctx_train == 262144
    schedule = deterministic_schedule(session)
    assert [item.label() for item in schedule] == [item.label() for item in full_schedule(session)] == [
        "baseline", "kv-q8_0-q8_0", "ctx-5888", "ctx-7168", "ctx-8960"]
    assert session.budgets.max_candidates == len(schedule) == 5
    baseline = baseline_proposal(session)
    fills = {}
    for proposal in schedule:
        config = run_config_for(session, None, proposal, baseline=baseline)
        fills[config.engine.ctx_size] = config.requested_input_tokens
    assert fills == {4864: 4096, 5888: 5120, 7168: 6400, 8960: 8192}  # the ceiling target really runs
    budgets = session.budgets  # the wall covers every candidate the cap allows beside the reserves
    assert budgets.wall_seconds >= (budgets.max_candidates * budgets.candidate_wall_seconds
                                    + budgets.holdout_reserve_seconds + budgets.cleanup_reserve_seconds)
    assert budgets.candidate_wall_seconds > session.base.bounds.minimum_wall_seconds()


def test_tune_checks_the_declared_training_context_against_the_weights(tmp_path, isolated_gpu_lock):
    """REV-LIVE-05: cover the call site in ``tune``, not only check_context_range_against_headers itself."""
    session = make_session(tmp_path, quantizations=("Q4_K_M",), spec_types=("none",), reasoning=("off",),
                           kv_pairs=(("f16", "f16"),), ctx_tiers=(2816, 6912), context_floor=2048,
                           context_ceiling=6000, context_n_ctx_train=262144)
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session))

    def weaker_weights(path):  # the GGUF headers declare less than the session's context range was checked against
        return {**metadata_reader()(path), "n_ctx_train": 32768}

    refused = tmp_path / "refused"
    with pytest.raises(ValueError, match="the weights do not have"):
        tune(session, refused, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
             metadata_reader=weaker_weights)
    assert runner.calls == [] and not (refused / "session.json").exists()  # refused before any candidate ran
    outcome = tune(session, tmp_path / "out", runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
                   metadata_reader=metadata_reader())  # headers that agree: the same session runs
    assert development_labels(runner) == ["baseline", "ctx-6912"]
    assert outcome["stop_reason"] == "complete"


def test_a_visible_gpu_lock_refuses_a_new_session_before_anything_is_written(tmp_path, isolated_gpu_lock):
    """A launch that died on the lock used to leave a ledger behind; the retry was then told to `resume` a
    session that had never started. The refusal now comes first and names the holder."""
    session = make_session(tmp_path)
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session))
    lock = isolated_gpu_lock / "llmbench-gpu-resource.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "created": "2026-09-20T21:42:41+00:00"}), encoding="utf-8")
    output = tmp_path / "blocked"
    with pytest.raises(FileExistsError, match=r"the GPU is in use: .*held by pid \d+"):
        tune(session, output, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
             metadata_reader=metadata_reader())
    assert runner.calls == [] and not output.exists() and lock.exists()
    lock.unlink()  # once the holder is gone the very same output directory starts normally
    tune(session, output, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
         metadata_reader=metadata_reader())
    assert runner.calls and (output / "session.json").exists()


def test_holdout_seed_must_differ_from_development_seeds(tmp_path):
    with pytest.raises(ValidationError, match="holdout NIAH seed"):
        make_session(tmp_path, holdout_seed=42)
    assert make_session(tmp_path / "ok", holdout_seed=43).holdout.niah_seed == 43


def test_policy_for_maps_budgets_and_live_mode(tmp_path):
    session = make_session(tmp_path)
    policy = policy_for(session)
    assert policy.mode == RunMode.LIVE and policy.name == "session-test"
    assert policy.budgets.wall_seconds == 3600 and policy.budgets.reserve_validation_seconds == 900
    assert policy.budgets.task_timeout_seconds == 900 and policy.budgets.max_candidates == 4
    assert policy.budgets.max_validation_candidates == 2
    assert policy.acceptance.speed_repetitions == session.base.speed.repetitions == 3
    assert policy.acceptance.minimum_tokens_per_second == 50


def test_run_config_for_fills_images_from_bundle_and_keeps_fingerprint_stable(tmp_path):
    session = make_session(tmp_path)
    bundle = make_bundle()
    baseline = baseline_proposal(session)
    first = run_config_for(session, bundle, baseline)
    again = run_config_for(session, bundle, baseline)
    assert first.fingerprint() == again.fingerprint()
    assert first.inference_image == bundle.inference and first.evaluator.mode == "host-process"
    assert first.evaluator.image is None and first.worker_image is None  # host-process stays image-less
    assert first.label == "baseline" and first.session_id == "session-test"
    assert first.bounds.candidate_wall_seconds == 900 and first.engine.ctx_size == 8192
    assert first.engine.cache_type_k == "f16" and first.engine.spec_type == "none" and first.engine.reasoning == "off"
    without = run_config_for(session, None, baseline)
    assert without.inference_image == session.base.inference_image and without.fingerprint() != first.fingerprint()
    kv = run_config_for(session, bundle, Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0")}), baseline=baseline)
    assert (kv.engine.cache_type_k, kv.engine.cache_type_v) == ("q8_0", "q8_0") and kv.label == "kv-q8_0-q8_0"
    assert kv.fingerprint() != first.fingerprint()
    context = run_config_for(session, bundle, Proposal(family="context", changes={"ctx_tier": 32768}), baseline=baseline)
    assert context.engine.ctx_size == 32768 and context.requested_input_tokens == 16384  # base fill ratio kept
    with pytest.raises(ValueError, match="baseline"):
        run_config_for(session, bundle, Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0")}))


def _manifest(config, template=TEMPLATE):
    return to_manifest(config, template_hash=template_hash(template), scorer_revision="s" * 8, environment_hash="e" * 8)


def test_to_manifest_maps_every_family_field_and_hashes_unmapped_into_runtime_revision(tmp_path):
    session = make_session(tmp_path)
    baseline = baseline_proposal(session)
    config = run_config_for(session, None, Proposal(family="speculation", changes={"spec_type": "draft-mtp"}),
                            baseline=baseline)
    manifest = _manifest(config)
    backend, model = manifest.backend, manifest.model
    assert backend.engine == "llama.cpp" and backend.context_length == 8192 and backend.gpu_offload == 1.0
    assert (backend.k_cache, backend.v_cache, backend.kv_on_gpu, backend.flash_attention) == ("f16", "f16", True, True)
    assert backend.mtp is True and backend.parallel == 1 and backend.batch_size == 2048 and backend.ubatch_size == 512
    assert backend.cpu_threads == 8 and backend.reasoning == "off"
    assert backend.runtime_revision.startswith("b11011-aa39d7a3e+") and len(backend.runtime_revision.split("+")[1]) == 12
    assert model.model_key == "Qwen3.8-27B:Q4_K_M" and model.quantization == "Q4_K_M"
    assert model.sha256 == config.model.sha256 and model.provenance["size_bytes"] == config.model.size_bytes
    assert manifest.mode == RunMode.LIVE and manifest.requested_input_tokens == 4096
    assert [item.suite for item in manifest.tasks] == ["tool-probes", "niah"]
    assert manifest.annotations["label"] == "spec-draft-mtp" and manifest.annotations["session_id"] == "session-test"
    # Unmapped engine fields change runtime_revision and nothing else.
    raw = config.model_dump(mode="json")
    raw["engine"]["threads_batch"] = 4
    other = _manifest(ContainerRunConfig.model_validate_json(canonical_json(raw)))
    assert other.backend.runtime_revision != backend.runtime_revision
    assert (other.model_dump(mode="json", exclude={"backend", "annotations"})
            == manifest.model_dump(mode="json", exclude={"backend", "annotations"}))
    assert other.backend.model_dump(exclude={"runtime_revision"}) == backend.model_dump(exclude={"runtime_revision"})
    # A numeric layer count needs the block count to express gpu_offload; guessing is refused.
    raw["engine"]["n_gpu_layers"] = 24
    partial = ContainerRunConfig.model_validate_json(canonical_json(raw))
    with pytest.raises(ValueError, match="block_count"):
        _manifest(partial)
    assert to_manifest(partial, template_hash="sha256:" + "0" * 64, scorer_revision="s", environment_hash="e",
                       block_count=48).backend.gpu_offload == 0.5


def test_manifests_differing_only_in_unmapped_engine_fields_are_not_comparable(tmp_path):
    session = make_session(tmp_path)
    baseline = run_config_for(session, None, baseline_proposal(session))
    raw = baseline.model_dump(mode="json")
    raw["engine"]["cache_prompt"] = True
    changed = ContainerRunConfig.model_validate_json(canonical_json(raw))
    with pytest.raises(ValueError, match="exactly one declared family"):
        infer_comparison_family(_manifest(baseline), _manifest(changed))
    kv = run_config_for(session, None, Proposal(family="kv", changes={"kv_pair": ("q8_0", "q8_0")}),
                        baseline=baseline_proposal(session))
    assert infer_comparison_family(_manifest(baseline), _manifest(kv)) == "kv"
    reasoning = run_config_for(session, None, Proposal(family="reasoning", changes={"reasoning": "on"}),
                               baseline=baseline_proposal(session))
    assert infer_comparison_family(_manifest(baseline), _manifest(reasoning)) == "reasoning"
    assert infer_comparison_family(_manifest(baseline), _manifest(baseline)) == "reference"


def test_to_manifest_context_fields_match_sample_context_identity(tmp_path):
    session = make_session(tmp_path)
    config = run_config_for(session, None, baseline_proposal(session))
    pilot_template = json.loads((Path(__file__).parent / "data" / "props-b11011.json").read_text(
        encoding="utf-8"))["body"]["chat_template"]
    manifest = to_manifest(config, template_hash=template_hash(pilot_template), scorer_revision="s",
                           environment_hash="e")
    # The evaluator labels every retrieval sample with these two identities (pilot evaluation.json).
    assert manifest.model.tokenizer_hash == "gguf-sha256:" + config.model.sha256
    assert manifest.model.template_hash == PILOT_TEMPLATE_ID
    sample = fake_evaluation(config, pilot_template)["samples"][-1]
    assert sample["context"]["tokenizer_id"] == manifest.model.tokenizer_hash
    assert sample["context"]["template_id"] == manifest.model.template_hash
    assert sample["context"]["actual_input_tokens"] <= manifest.requested_input_tokens
    assert sample["context"]["actual_input_tokens"] + manifest.generation.max_output_tokens <= manifest.backend.context_length


# LIVE-013: the exact RULER selection the ruler-kv-1 campaign declared in its session base.
LIVE_RULER_BENCHMARK = {
    "benchmark_id": "ruler", "revision": "ruler-vendored-v1", "split": "development", "seed": 42,
    "task_ids": [f"ruler/{task}/{length}" for length in (16384, 32768, 65536, 131072)
                 for task in ("niah_multikey_3", "vt")],
    "options": {"tasks": ["niah_multikey_3", "vt"], "lengths": [16384, 32768, 65536, 131072],
                "ruler_output_tokens": 512}}


def live_ruler_selection():
    return BenchmarkSelection.model_validate_json(canonical_json(LIVE_RULER_BENCHMARK))


def test_to_manifest_carries_benchmark_options_into_the_manifest(tmp_path):
    """LIVE-013: a session declaring ``ruler_output_tokens: 512`` must record it in the manifest.

    Before this, ``BenchmarkSelection.to_task_selection`` dropped options, the manifest carried none and the
    executor rebuilt every candidate without them: RULER fell back to upstream's 30-token ``vt`` cap and the
    truncation was scored as a retrieval failure (artifacts/container-campaign/ruler-kv-1).
    """
    session = make_session(tmp_path)
    config = run_config_for(session, None, baseline_proposal(session), benchmarks=(live_ruler_selection(),))
    manifest = _manifest(config)
    assert [item.suite for item in manifest.tasks] == ["ruler"]
    assert dict(manifest.tasks[0].options) == {"tasks": ("niah_multikey_3", "vt"),
                                               "lengths": (16384, 32768, 65536, 131072),
                                               "ruler_output_tokens": 512}
    # Byte-for-byte with what the session declared, through the JSON the manifest is stored as.
    assert (canonical_json(manifest.tasks[0].options)
            == canonical_json(config.benchmarks[0].options)
            == canonical_json(LIVE_RULER_BENCHMARK["options"]))
    reread = ExperimentManifest.model_validate_json(manifest.model_dump_json())
    assert reread.tasks[0].options == manifest.tasks[0].options and reread.fingerprint() == manifest.fingerprint()


def test_to_manifest_without_options_is_unchanged_and_option_changes_are_a_new_experiment(tmp_path):
    """No-options sessions keep their exact manifest identity; an option change is a different experiment."""
    session = make_session(tmp_path)
    config = run_config_for(session, None, baseline_proposal(session))
    manifest = _manifest(config)
    assert [dict(item.options) for item in manifest.tasks] == [{}, {}]
    # The manifest JSON a pre-LIVE-013 run recorded still validates and still hashes identically.
    recorded = json.loads(manifest.model_dump_json())
    for task in recorded["tasks"]:
        del task["options"]
    assert ExperimentManifest.model_validate_json(canonical_json(recorded)).fingerprint() == manifest.fingerprint()
    raw = json.loads(canonical_json(LIVE_RULER_BENCHMARK))
    capped = _manifest(run_config_for(session, None, baseline_proposal(session),
                                      benchmarks=(live_ruler_selection(),)))
    raw["options"]["ruler_output_tokens"] = 30
    uncapped = _manifest(run_config_for(session, None, baseline_proposal(session), benchmarks=(
        BenchmarkSelection.model_validate_json(canonical_json(raw)),)))
    assert capped.fingerprint() != uncapped.fingerprint() != manifest.fingerprint()
    assert capped.backend == uncapped.backend and capped.generation == uncapped.generation


def test_to_manifest_refuses_options_it_cannot_carry(tmp_path):
    """Belt and braces: a selection whose options will not survive the trip aborts, it never runs silently."""
    from llmbench.containers.campaign import BenchmarkOptionsNotCarried

    class Unserialisable:
        benchmark_id = "ruler"
        options = {"lengths": {16384, 32768}}  # a set: JSON cannot carry it

        def to_task_selection(self):
            return live_ruler_selection().to_task_selection()

    session = make_session(tmp_path)
    config = run_config_for(session, None, baseline_proposal(session), benchmarks=(live_ruler_selection(),))
    broken = config.model_copy(update={"benchmarks": (Unserialisable(),)})
    with pytest.raises(BenchmarkOptionsNotCarried, match="cannot be carried into the manifest"):
        _manifest(broken)


def test_example_session_validates_and_matches_the_live_acceptance_plan():
    """examples/session.json is the C-L step-1 input (plan 4.9): two kv candidates at 8K."""
    from llmbench.containers.proposals import deterministic_schedule
    from llmbench.containers.session import read_session_config
    session = read_session_config(EXAMPLE.with_name("session.json"))
    assert session.session_id == "q4-kv-8k" and session.image_bundle is None
    assert session.assets[0].sha256 == json.loads(EXAMPLE.read_text(encoding="utf-8"))["assets"][0]["sha256"]
    assert session.search.kv_pairs == (("f16", "f16"), ("q8_0", "q8_0")) and session.search.ctx_tiers == (8192,)
    assert session.holdout.niah_seed == 7 and {item.seed for item in session.base.benchmarks} == {42}
    assert (session.budgets.wall_seconds, session.budgets.max_candidates, session.budgets.holdout_reserve_seconds,
            session.budgets.candidate_wall_seconds) == (2400, 2, 900, 900)
    assert [item.label() for item in deterministic_schedule(session)] == ["baseline", "kv-q8_0-q8_0"]
    policy = policy_for(session)
    assert policy.budgets.wall_seconds == 2400 and policy.budgets.reserve_validation_seconds == 900


# ---- metal-native runtime -------------------------------------------------------------------------------------
# The session machinery is runtime-agnostic: a native bundle pins a host llama-server where an image bundle pins
# a CUDA image, and every NVIDIA path above must stay exactly what it was.

NATIVE_LOCK = SessionLock(allow_model_operations=True, allow_inference=True, allow_native_execution=True)
"""A policy for the Mac: native execution, no containers. ``LOCK`` above is the NVIDIA policy (no native)."""
ARM64_WORKER = {"role": "worker", "reference": "sha256:" + "7" * 64, "image_id": "sha256:" + "7" * 64,
                "platform": "linux/arm64", "entrypoint": ["python", "-m", "llmbench.coding.worker"]}


def make_native_bundle(*, executable="/Users/example/llama-b11011/llama-server", executable_sha256="d" * 64,
                       libraries_sha256="e" * 64, sandbox=None, worker=None):
    """A NativeBundle as `prepare --runtime metal-native` records it (no process is ever started from it here)."""
    from llmbench.containers.config import NativeBundle
    server = {"executable": executable, "executable_sha256": executable_sha256,
              "libraries_sha256": libraries_sha256, "build_info": "b11011-aa39d7a3e", "help_sha256": "f" * 64,
              "source": "llama.cpp b11011 macos-arm64 release"}
    return NativeBundle.model_validate_json(canonical_json({
        "prepared_utc": "2026-09-23T00:00:00+00:00", "native_server": server,
        "libraries": {"libggml-metal.dylib": "1" * 64}, "worker": worker, "help_sha256": "f" * 64,
        "registry_digest": "c" * 64, "host": {"machine": "arm64", "chip": "Apple M1"},
        "build": {"provenance": "unrecorded"},
        "sandbox": {"status": "blocked", "reason": "docker CLI not found"} if sandbox is None else sandbox}))


def native_session(tmp_path, bundle=None, *, native_limits=None, **options) -> ContainerSessionConfig:
    """``make_session`` with its base served by a native bundle, as a derived metal-native session would be."""
    from llmbench.containers.session import native_overlay
    raw = json.loads(make_session(tmp_path, **options).model_dump_json())
    native_overlay(raw["base"], bundle or make_native_bundle())
    if native_limits is not None:
        raw["base"]["native_limits"] = native_limits
    return ContainerSessionConfig.model_validate_json(canonical_json(raw))


def test_a_native_bundle_turns_candidates_into_metal_runs_and_pins_the_server_not_its_location(tmp_path):
    from llmbench.containers.config import NativeLimits
    from llmbench.containers.session import image_ids
    session = make_session(tmp_path)  # an NVIDIA session: the bundle alone decides what serves its candidates
    bundle = make_native_bundle(worker=ARM64_WORKER)
    baseline = baseline_proposal(session)
    config = run_config_for(session, bundle, baseline)
    assert config.runtime == "metal-native" and config.inference_image is None
    assert config.native_server == bundle.native_server and config.native_limits == NativeLimits()
    assert config.worker_image == bundle.worker and config.worker_image.platform == "linux/arm64"
    assert config.evaluator.mode == "host-process" and config.server_model_path == config.model.host_path
    # The ledger keeps its `image_ids` key; a native bundle's pins have disjoint keys, so the two kinds of bundle
    # can never compare equal on resume, and an image bundle still pins exactly what it always did.
    assert image_ids(bundle) == {"native_server": "d" * 64, "libraries": "e" * 64, "worker": "sha256:" + "7" * 64}
    assert image_ids(make_bundle()) == {"inference": "sha256:" + "a" * 64, "evaluator": "sha256:" + "b" * 64,
                                        "worker": None}
    # Where the executable lives is not identity; what it is (its digests) is.
    moved = run_config_for(session, make_native_bundle(executable="/opt/elsewhere/llama-server",
                                                       worker=ARM64_WORKER), baseline)
    assert moved.fingerprint() == config.fingerprint()
    assert moved.native_server.executable == "/opt/elsewhere/llama-server"
    rebuilt = run_config_for(session, make_native_bundle(libraries_sha256="9" * 64, worker=ARM64_WORKER), baseline)
    assert rebuilt.fingerprint() != config.fingerprint()
    # Limits a native session was tuned under survive a bundle; an image bundle cannot serve a native session.
    tuned = native_session(tmp_path / "n", native_limits={"memory_reserve_mib": 2048, "max_swap_growth_mib": 512})
    kept = run_config_for(tuned, make_native_bundle(executable="/opt/moved/llama-server"), baseline_proposal(tuned))
    assert (kept.native_limits.memory_reserve_mib, kept.native_limits.max_swap_growth_mib) == (2048, 512)
    with pytest.raises(ValueError, match="native-bundle.json"):
        run_config_for(tuned, make_bundle(), baseline_proposal(tuned))


def test_the_metal_backend_of_a_commit_is_never_the_cuda_backend_of_the_same_commit(tmp_path):
    session = make_session(tmp_path)
    baseline = baseline_proposal(session)
    cuda = _manifest(run_config_for(session, None, baseline))
    metal = _manifest(run_config_for(session, make_native_bundle(), baseline))
    assert cuda.backend.runtime_revision.startswith("b11011-aa39d7a3e+")  # NVIDIA: unchanged format
    assert metal.backend.runtime_revision == "metal:" + cuda.backend.runtime_revision  # same build, same engine
    # Only the revision differs -- no BackendSettings field was added -- so the controller can never read a
    # Metal-vs-CUDA pair as a single-family treatment of one backend.
    assert metal.backend.model_dump(exclude={"runtime_revision"}) == cuda.backend.model_dump(
        exclude={"runtime_revision"})
    with pytest.raises(ValueError, match="exactly one declared family"):
        infer_comparison_family(cuda, metal)
    assert metal.model.model_dump() == cuda.model.model_dump() and metal.tasks == cuda.tasks


def test_planning_slowdown_is_recorded_only_when_it_is_not_the_nvidia_default(tmp_path):
    plain = make_session(tmp_path)
    assert plain.planning_slowdown == 1.0
    assert "planning_slowdown" not in plain.model_dump(mode="json")
    assert "planning_slowdown" not in json.loads(plain.model_dump_json())  # session-config.json bytes unchanged
    raw = json.loads(plain.model_dump_json())
    raw["planning_slowdown"] = 3.0
    slow = ContainerSessionConfig.model_validate_json(json.dumps(raw))
    assert slow.planning_slowdown == 3.0 and slow.model_dump(mode="json")["planning_slowdown"] == 3.0
    assert ContainerSessionConfig.model_validate_json(slow.model_dump_json()) == slow
    for bad in (0, -1, 101):
        with pytest.raises(ValidationError):
            ContainerSessionConfig.model_validate_json(json.dumps({**raw, "planning_slowdown": bad}))


def test_a_native_session_needs_native_permission_and_resume_refuses_a_different_server(tmp_path,
                                                                                    isolated_gpu_lock):
    from llmbench.containers.session import read_ledger, resume
    from llmbench.safety import OperationForbidden
    bundle = make_native_bundle()
    session = native_session(tmp_path, bundle)
    bundle_path = tmp_path / "prep" / "native-bundle.json"
    bundle_path.parent.mkdir()
    bundle_path.write_text(bundle.model_dump_json(), encoding="utf-8")
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session))
    refused = tmp_path / "refused"
    with pytest.raises(OperationForbidden, match="native"):  # an NVIDIA policy never authorizes a host binary
        tune(session, refused, runner=runner, session_lock=LOCK, clock=clock, wall=clock.wall,
             metadata_reader=metadata_reader(), bundle_path=bundle_path)
    assert runner.calls == [] and not refused.exists()
    # The Mac policy runs it without any container permission (no broker, so no sandbox workers either).
    out = tmp_path / "out"
    outcome = tune(session, out, runner=runner, session_lock=NATIVE_LOCK, clock=clock, wall=clock.wall,
                   metadata_reader=metadata_reader(), bundle_path=bundle_path)
    assert outcome["stop_reason"] in {"complete", "candidates"} and runner.calls
    assert {call["config"].runtime for call in runner.calls} == {"metal-native"}
    assert {call["config"].native_server.executable_sha256 for call in runner.calls} == {"d" * 64}
    ledger = read_ledger(out)
    assert ledger["image_bundle"] == str(bundle_path.resolve())
    assert ledger["image_ids"] == {"native_server": "d" * 64, "libraries": "e" * 64, "worker": None}
    # resume: a rebuilt llama-server (different library digest) is refused before anything runs ...
    second = FakeRunner(clock, policy=policy_for(session))
    with pytest.raises(ValueError, match="differ from the ledger"):
        resume(out, runner=second, session_lock=NATIVE_LOCK, clock=clock, wall=clock.wall,
               bundle=make_native_bundle(libraries_sha256="9" * 64))
    with pytest.raises(ValueError, match="differ from the ledger"):  # ... and so is an image bundle
        resume(out, runner=second, session_lock=NATIVE_LOCK, clock=clock, wall=clock.wall, bundle=make_bundle())
    with pytest.raises(OperationForbidden):  # the policy is checked for the session's own runtime
        resume(out, runner=second, session_lock=LOCK, clock=clock, wall=clock.wall)
    # ... while the same server moved elsewhere is the same session, and its bundle missing names the flag to use.
    moved = resume(out, runner=second, session_lock=NATIVE_LOCK, clock=clock, wall=clock.wall,
                   bundle=make_native_bundle(executable="/opt/moved/llama-server"))
    assert second.calls == [] and moved["stop_reason"] in {"complete", "candidates"}
    bundle_path.unlink()
    with pytest.raises(ValueError, match="pass the moved native-bundle.json as --image-bundle") as missing:
        resume(out, runner=second, session_lock=NATIVE_LOCK, clock=clock, wall=clock.wall)
    assert "--native-bundle" not in str(missing.value)  # the resume parser has no such flag
    # The advice works as given: the resume command line accepts the moved native bundle through --image-bundle.
    from llmbench.containers.session import main
    relocated = tmp_path / "moved" / "native-bundle.json"
    relocated.parent.mkdir()
    relocated.write_text(make_native_bundle(executable="/opt/moved/llama-server").model_dump_json(), encoding="utf-8")
    mac = _policy_file(tmp_path, "mac-policy.json", allow_native_execution=True)
    assert main(["resume", "--output", str(out), "--image-bundle", str(relocated), "--policy", str(mac)],
                runner_factory=lambda args, policy: second) == 0
    assert second.calls == []  # every candidate had already run; nothing was re-measured


def test_read_bundle_tells_the_two_bundle_kinds_apart_by_content(tmp_path):
    from llmbench.containers.config import ImageBundle, NativeBundle
    from llmbench.containers.session import read_bundle
    native, image = tmp_path / "image-bundle.json", tmp_path / "native-bundle.json"  # names deliberately swapped
    native.write_text(make_native_bundle().model_dump_json(), encoding="utf-8")
    image.write_text(make_bundle().model_dump_json(), encoding="utf-8")
    assert isinstance(read_bundle(native), NativeBundle) and isinstance(read_bundle(image), ImageBundle)
    hybrid = json.loads(make_bundle().model_dump_json())
    hybrid["native_server"] = json.loads(make_native_bundle().model_dump_json())["native_server"]
    (tmp_path / "hybrid.json").write_text(json.dumps(hybrid), encoding="utf-8")
    with pytest.raises(ValidationError):  # neither strict model accepts the other's keys
        read_bundle(tmp_path / "hybrid.json")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON bundle"):
        read_bundle(tmp_path / "broken.json")
    with pytest.raises(ValueError, match="regular file"):
        read_bundle(tmp_path / "absent.json")


def _tune_args(tmp_path, weights, output, policy, **overrides):
    import argparse
    values = dict(config=None, model=str(weights), output=str(output), budget_seconds=3600, image_bundle=None,
                  search="default", context_floor=None, context_ceiling=None, base_config=str(EXAMPLE),
                  policy=str(policy), capabilities_dir="artifacts/container-prep", runtime=None, native_bundle=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def _policy_file(tmp_path, name, **grants):
    path = tmp_path / name
    path.write_text(json.dumps({"allow_model_operations": True, "allow_inference": True, **grants}), encoding="utf-8")
    return path


def test_main_tune_derives_a_metal_session_from_the_native_bundle_and_refuses_contradictions(
        tmp_path, capsys, isolated_gpu_lock, monkeypatch):
    from llmbench.containers.session import SESSION_CONFIG_NAME, main_tune, read_ledger, read_session_config
    monkeypatch.chdir(tmp_path)  # no staged datasets: the plan's content is not what this test is about
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "tiny-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M", name="tiny-model", n_ctx_train=32768, nextn=0))
    bundle_path = tmp_path / "native-bundle.json"
    bundle_path.write_text(make_native_bundle().model_dump_json(), encoding="utf-8")
    mac = _policy_file(tmp_path, "mac-policy.json", allow_native_execution=True)
    nvidia = _policy_file(tmp_path, "nvidia-policy.json", allow_container_execution=True)
    clock, runners = FakeClock(), []

    def factory(args, campaign_policy):
        runners.append(FakeRunner(clock, policy=campaign_policy))
        return runners[-1]

    def run(output, policy=mac, **overrides):
        return main_tune(_tune_args(tmp_path, weights, output, policy, **overrides), runner_factory=factory,
                         clock=clock, wall=clock.wall)

    native = dict(runtime="metal-native", native_bundle=str(bundle_path))
    for overrides, message in (({"runtime": "metal-native"}, "needs --native-bundle"),
                               ({"native_bundle": str(bundle_path)}, "add --runtime metal-native"),
                               ({"runtime": "nvidia-container", "native_bundle": str(bundle_path)},
                                "add --runtime metal-native"),
                               ({**native, "image_bundle": str(bundle_path)}, "pinned by --native-bundle alone"),
                               ({"image_bundle": str(bundle_path)}, "is a native bundle")):
        assert run(tmp_path / "refused", policy=nvidia if "image_bundle" in overrides and len(overrides) == 1
                   else mac, **overrides) == 2
        assert message in capsys.readouterr().err and not (tmp_path / "refused").exists()
    assert run(tmp_path / "denied", policy=nvidia, **native) == 2  # container permission is not native permission
    assert "native forbidden" in capsys.readouterr().err and not (tmp_path / "denied").exists()
    assert runners == []
    output = tmp_path / "s1"
    assert run(output, **native) == 0
    capsys.readouterr()
    session = read_session_config(output / SESSION_CONFIG_NAME)
    assert session.base.runtime == "metal-native" and session.base.inference_image is None
    assert session.planning_slowdown == 3.0 and session.image_bundle == str(bundle_path)
    assert session.search.ctx_tiers == (4864,)  # RULER's shortest length, output and template reserved on top
    assert (session.search.context_floor, session.search.context_ceiling) == (4096, 4096)
    assert read_ledger(output)["image_ids"]["native_server"] == "d" * 64
    assert {call["config"].runtime for call in runners[0].calls} == {"metal-native"}
    # A --config session states its own runtime; the flags that derive one are refused beside it.
    assert run(tmp_path / "s2", config=str(output / SESSION_CONFIG_NAME), model=None,
               native_bundle=str(bundle_path)) == 2
    assert "--native-bundle derives" in capsys.readouterr().err
    assert run(tmp_path / "s3", config=str(output / SESSION_CONFIG_NAME), model=None,
               runtime="nvidia-container") == 2
    assert "differs from the metal-native runtime" in capsys.readouterr().err
    assert run(tmp_path / "s4", policy=nvidia, config=str(output / SESSION_CONFIG_NAME), model=None) == 2
    assert not (tmp_path / "s4").exists() and len(runners) == 1


def test_main_tune_authorizes_the_broker_only_when_the_sandbox_verdict_keeps_it(tmp_path, capsys, isolated_gpu_lock,
                                                                                 monkeypatch):
    """A Mac policy without container permission is a reason prepare records the sandbox as blocked; the derived
    session then drops the template's broker, so tune must not demand container permission for it up front. When
    the sandbox is available the broker stays and starts Docker workers, so container permission is required."""
    from llmbench.containers.config import read_run_config
    from llmbench.containers.derive import planned_operations
    from llmbench.containers.session import NVIDIA_OPERATIONS, SESSION_CONFIG_NAME, main_tune, read_session_config
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "tiny-Q4_K_M.gguf").write_bytes(gguf_bytes("Q4_K_M", name="tiny-model", n_ctx_train=32768, nextn=0))
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["broker"] = {"max_requests": 2}
    base_config = tmp_path / "broker-base.json"
    base_config.write_text(json.dumps(raw), encoding="utf-8")
    blocked, available = tmp_path / "blocked.json", tmp_path / "available.json"
    blocked.write_text(make_native_bundle().model_dump_json(), encoding="utf-8")  # docker CLI not found
    available.write_text(make_native_bundle(sandbox={"status": "available", "reason": None},
                                            worker=ARM64_WORKER).model_dump_json(), encoding="utf-8")
    base = read_run_config(base_config)
    assert planned_operations(base, native_bundle=make_native_bundle()) == ("native", "load", "inference")
    assert planned_operations(base, native_bundle=make_native_bundle(sandbox={"status": "available"},
                                                                     worker=ARM64_WORKER)) == (
        "native", "load", "inference", "container")
    assert planned_operations(base) == NVIDIA_OPERATIONS  # no native bundle: the NVIDIA tuple, broker or not
    mac = _policy_file(tmp_path, "mac-policy.json", allow_native_execution=True)
    clock, runners = FakeClock(), []

    def factory(args, campaign_policy):
        runners.append(FakeRunner(clock, policy=campaign_policy))
        return runners[-1]

    def run(output, bundle):
        return main_tune(_tune_args(tmp_path, weights, output, mac, base_config=str(base_config),
                                    runtime="metal-native", native_bundle=str(bundle)),
                         runner_factory=factory, clock=clock, wall=clock.wall)

    assert run(tmp_path / "needs-container", available) == 2
    assert "container forbidden" in capsys.readouterr().err
    assert not (tmp_path / "needs-container").exists() and runners == []
    assert run(tmp_path / "blocked", blocked) == 0
    session = read_session_config(tmp_path / "blocked" / SESSION_CONFIG_NAME)
    assert session.base.broker is None and "coding" not in {item.benchmark_id for item in session.base.benchmarks}
    assert {call["config"].broker for call in runners[0].calls} == {None}


def test_the_default_session_runner_dispatches_on_runtime_and_builds_nothing_up_front(tmp_path, monkeypatch):
    import argparse
    from llmbench.containers import runner as runner_module
    from llmbench.containers.runtime import DispatchRunner
    from llmbench.containers.session import _default_runner_factory
    session = make_session(tmp_path)
    policy = policy_for(session)
    runner = _default_runner_factory(argparse.Namespace(capabilities_dir="caps", policy="p.json"), policy)
    assert isinstance(runner, DispatchRunner) and runner.synthetic is False
    assert (runner.capabilities_dir, runner.policy_path, runner.policy) == ("caps", "p.json", policy)
    assert runner.built == ()  # neither ContainerRunner nor NativeRunner exists until a candidate arrives
    # An NVIDIA candidate reaches exactly the ContainerRunner this factory built before runtimes existed: the same
    # three keywords, built once and reused, and each run's arguments passed through untouched.
    built, runs = [], []

    class RecordingContainerRunner:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def run(self, config, output_dir, **kwargs):
            runs.append((config, output_dir, kwargs))
            return "result"

    monkeypatch.setattr(runner_module, "ContainerRunner", RecordingContainerRunner)
    config = run_config_for(session, None, baseline_proposal(session))
    assert runner.run(config, tmp_path / "a", remaining_budget_seconds=5.0, lease_held=True) == "result"
    assert runner.run(config, tmp_path / "b") == "result"
    assert built == [{"capabilities_dir": "caps", "policy_path": "p.json", "policy": policy}]
    assert runs == [(config, tmp_path / "a", {"remaining_budget_seconds": 5.0, "lease_held": True}),
                    (config, tmp_path / "b", {"remaining_budget_seconds": None, "lease_held": False})]
    assert runner.built == ("nvidia-container",)  # no NativeRunner was imported or built for an NVIDIA session


def test_nvidia_sessions_still_need_exactly_the_container_policy(tmp_path, isolated_gpu_lock):
    from llmbench.containers.session import NVIDIA_OPERATIONS, authorize_session, required_operations_for
    from llmbench.safety import OperationForbidden
    session = make_session(tmp_path)
    assert required_operations_for(session.base) == NVIDIA_OPERATIONS == ("container", "load", "inference")
    authorize_session(LOCK, session, make_bundle())  # the NVIDIA policy, NVIDIA bundle: allowed as always
    clock = FakeClock()
    runner = FakeRunner(clock, policy=policy_for(session))
    with pytest.raises(OperationForbidden, match="container"):  # native permission never authorizes containers
        tune(session, tmp_path / "out", runner=runner, session_lock=NATIVE_LOCK, clock=clock, wall=clock.wall,
             metadata_reader=metadata_reader())
    assert runner.calls == [] and not (tmp_path / "out").exists()
    # A native bundle over an NVIDIA session changes what runs, so it changes what must be allowed.
    with pytest.raises(OperationForbidden, match="native"):
        authorize_session(LOCK, session, make_native_bundle())
    # A native session with a coding broker starts sandboxed Docker workers: container permission as well.
    raw = json.loads(native_session(tmp_path / "b").model_dump_json())
    raw["base"]["broker"] = {"max_requests": 2}
    brokered = ContainerSessionConfig.model_validate_json(json.dumps(raw))
    assert required_operations_for(brokered.base) == ("native", "load", "inference", "container")
    with pytest.raises(OperationForbidden, match="container"):
        authorize_session(NATIVE_LOCK, brokered)
