"""Runtime-aware reports: a metal-native candidate is reported with unified-memory evidence and never as VRAM, a
sandbox-blocked coding suite is reported as blocked and never as a zero, and every NVIDIA report reads exactly as it
did before runtimes existed. Pure rendering against fakes and hand-built run directories; nothing runs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from llmbench.config import canonical_json
from llmbench.containers.config import ContainerRunConfig, ContainerRunResult
from llmbench.containers.report import (CANDIDATE_UNIFIED_NOTE, SANDBOX_BLOCKED, candidate_markdown, default_policy,
                                        evidence_row, summarize_evaluation, write_candidate_reports)
from llmbench.containers.session_report import (COLUMNS, HEADERS, UNIFIED_COLUMNS, UNIFIED_MEMORY_NOTE, attempt_row,
                                                attempt_runtime, has_native_rows, power_text, pressure_text,
                                                session_html, session_markdown, unified_memory_evidence)
from llmbench.containers.session_report import SANDBOX_BLOCKED as SESSION_SANDBOX_BLOCKED
from llmbench.store import Store
from test_containers_campaign import run_tune, small_session
from test_containers_session import isolated_gpu_lock

_ = isolated_gpu_lock  # fixture re-export for pytest collection

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "candidate.json"
MIB = 1024 * 1024


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cross_model_report = _script("cross_model_report")
coding_report = _script("coding_report")

# The shape `native.NativeRunner` seals into `result.memory`, with the figures captured on the M1 this runtime was
# built on (Qwen3-1.7B-Q4_K_M, b11011, full offload): 1905 MiB phys_footprint, 1631.19 MiB of Metal buffers in a
# 5461 MiB working set.
MEMORY = {
    "kind": "apple-unified", "note": "Apple Silicon unified memory ... None of these is VRAM.",
    "admission": {"kind": "apple-unified", "available_bytes": 6144 * MIB, "verdict": "admitted"},
    "after_load": {"server_phys_footprint_mib": 1905.0, "server_rss_mib": 952.5, "host_memory_total_mib": 8192.0,
                   "host_memory_available_mib": 6144.0, "swap_used_mib": 6800.0, "memory_pressure_level": 2,
                   "gpu_device_utilization_percent": 3, "gpu_in_use_system_memory_mib": 1700.0,
                   "power": {"power_source": "battery", "power_source_label": "Battery Power", "battery_percent": 73,
                             "battery_state": "discharging", "charging": False}, "errors": []},
    "server_log": {"metal_device": "Apple M1", "metal_budget_mib": 5461, "metal_resident_mib": 1631.19,
                   "host_resident_mib": 256.19, "offloaded_layers": 29, "total_layers": 29},
    "during_evaluation": {"kind": "apple-unified-watchdog", "peak_phys_footprint_bytes": 1997537280,
                          "min_available_bytes": 6000 * MIB, "max_pressure_level": 2,
                          "swap_growth_bytes": 128 * MIB, "violations": []},
}


def nvidia_config() -> ContainerRunConfig:
    return ContainerRunConfig.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))


def metal_config() -> ContainerRunConfig:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.pop("inference_image")
    raw["runtime"] = "metal-native"
    raw["native_server"] = {"executable": "/Users/example/llama-b11011/llama-server", "executable_sha256": "a" * 64,
                            "libraries_sha256": "b" * 64, "build_info": "b11011-aa39d7a3e", "help_sha256": "c" * 64,
                            "source": "llama-b11011-bin-macos-arm64.tar.gz"}
    raw["native_limits"] = {}
    return ContainerRunConfig.model_validate_json(canonical_json(raw))


def result_for(config, *, samples=(), memory=None, vram=None, state="completed", **changes) -> ContainerRunResult:
    summary = summarize_evaluation(config, {"samples": list(samples), "speed_observations": []}, None)
    fields = {"session_id": "s", "attempt_id": "a" * 32, "config_fingerprint": config.fingerprint(),
              "project_name": "llmbench-aaaaaaaaaaaa", "state": state, "synthetic": False,
              "started_utc": "2026-09-23T00:00:00Z", "finished_utc": "2026-09-23T00:01:00Z", "elapsed_seconds": 60.0,
              "budget_charged_seconds": 60.0, "speed": summary["speed"], "quality": summary["quality"],
              "load_seconds": 2.6, "vram_used_mib_after_load": vram, **changes}
    if config.runtime == "metal-native":
        fields.update(runtime="metal-native", memory=MEMORY if memory is None else memory)
    return ContainerRunResult(**fields)


def render(config, result) -> str:
    policy = default_policy(config)
    return candidate_markdown(config, result, evidence_row(config, result, policy), policy)


TOOLS = {"task_id": "tools/a", "category": "tools", "score": 1.0, "status": "completed"}


# ---- candidate report (report.py) ------------------------------------------------------------------------------


def test_an_nvidia_candidate_report_keeps_its_image_and_vram_lines_and_gains_nothing():
    config = nvidia_config()
    text = render(config, result_for(config, samples=[TOOLS], vram=21179.0))
    image = config.inference_image
    assert text.startswith(f"# Container candidate `{config.label}`\n\nMeasured llama.cpp container candidate. One "
                           "candidate never establishes a validated preset.\n")
    assert f"- Image: `{image.reference}` build {image.build_info}\n" in text
    assert "- Elapsed 60.0s; model load 2.6s; VRAM after load 21179 MiB\n" in text
    assert "- Cleanup verified: **False**; abort campaign: False\n" in text
    for native_only in ("Native (Metal)", "llama-server:", "Unified memory", "not applicable", "Settings this runtime",
                        CANDIDATE_UNIFIED_NOTE, SANDBOX_BLOCKED, "battery"):
        assert native_only not in text
    assert text.endswith("Context capacity, filled prompt tokens and tested usable context are different "
                         "measurements.\n")
    # An NVIDIA result with no VRAM reading still says "unmeasured": a missed measurement is not "not applicable".
    assert "VRAM after load unmeasured" in render(config, result_for(config, vram=None))


def test_a_metal_candidate_report_names_the_pinned_executable_and_its_unified_memory():
    config = metal_config()
    text = render(config, result_for(config, samples=[TOOLS]))
    assert text.startswith(f"# Native (Metal) candidate `{config.label}`\n\nMeasured llama.cpp native (Metal) "
                           "candidate.")
    assert ("- llama-server: `/Users/example/llama-b11011/llama-server` sha256 `" + "a" * 64 + "`, libraries `"
            + "b" * 64 + "`, build b11011-aa39d7a3e, source llama-b11011-bin-macos-arm64.tar.gz") in text
    assert "- Image:" not in text
    # No VRAM claim of any kind: there is no dedicated VRAM, so "unmeasured" would claim a missed measurement.
    assert "VRAM after load unmeasured" not in text
    assert "- Elapsed 60.0s; model load 2.6s; VRAM after load not applicable: unified memory\n" in text
    assert ("- Unified memory after load: server footprint 1905 MiB (RSS 952 MiB); Metal buffers 1631 MiB of 5461 "
            "MiB budget; host available 6144 MiB; swap used 6800 MiB (growth 128 MiB); pressure level 2 (warn); "
            "power: battery 73%\n") in text
    assert ("- During evaluation: peak server footprint 1905 MiB; lowest host available 6000 MiB; highest pressure "
            "level 2 (warn); watchdog violations: none\n") in text
    assert "- Measured on battery power:" in text
    # The Docker cgroup limits bound nothing for a host process, and the report says so rather than implying they did.
    section = text.split("## Settings this runtime does not enforce\n\n", 1)[1].split("\n\n", 1)[0]
    assert [line.split(" (", 1)[0] for line in section.splitlines()] == [
        "- limits.inference_memory_mib", "- limits.inference_cpus", "- limits.max_foreign_vram_mib"]
    assert CANDIDATE_UNIFIED_NOTE in text and "VRAM" not in CANDIDATE_UNIFIED_NOTE.replace("no VRAM figure", "")


def test_a_metal_candidate_that_never_loaded_reports_every_figure_unmeasured_and_invents_none():
    config = metal_config()
    early = {"kind": "apple-unified", "note": "n", "admission": {"verdict": "admitted"}}
    text = render(config, result_for(config, memory=early, state="rejected", failure_stage="admit",
                                     failure_reasons=("the GPU is already 97% busy",), load_seconds=None))
    assert ("- Unified memory after load: server footprint unmeasured (RSS unmeasured); Metal buffers unmeasured of "
            "unmeasured budget; host available unmeasured; swap used unmeasured (growth unmeasured); pressure level "
            "unmeasured; power: unmeasured\n") in text
    assert "During evaluation" not in text and "battery" not in text  # the watchdog never ran; no summary is made up
    assert "model load unmeasured; VRAM after load not applicable: unified memory" in text


def test_a_blocked_coding_sandbox_is_never_reported_as_a_coding_score():
    config = metal_config()
    backfilled = [{"task_id": f"evalplus/{i}", "category": "coding", "score": 0.0, "status": "environment_error",
                   "reason": "sandbox_unavailable: docker CLI not found"} for i in range(3)]
    memory = {**MEMORY, "coding_sandbox": {"status": "blocked", "reason": "docker CLI not found"}}
    result = result_for(config, samples=[TOOLS, *backfilled], memory=memory)
    assert result.quality["coding"]["score"] == 0.0  # the denominator rule turned three environment errors into 0.0
    text = render(config, result)
    assert f"| coding | 3 | {SANDBOX_BLOCKED} |" in text and "| coding | 3 | 0.00 |" not in text
    assert ("- Coding sandbox: blocked (docker CLI not found); the suites that execute generated code were not run, "
            "and generated code is never executed on the host") in text
    # An available sandbox changes nothing about the score cell.
    fine = {**MEMORY, "coding_sandbox": {"status": "available", "reason": None}}
    ran = render(config, result_for(config, samples=[TOOLS, *backfilled], memory=fine))
    assert "| coding | 3 | 0.00 |" in ran and "- Coding sandbox: available\n" in ran


def test_watchdog_violations_and_leftover_processes_are_printed_verbatim():
    config = metal_config()
    reason = "server phys_footprint 5600 MiB exceeds the 5461 MiB limit (metal_budget)"
    memory = copy.deepcopy(MEMORY)
    memory["during_evaluation"]["violations"] = [{"kind": "footprint", "reason": reason, "monotonic_seconds": 1.0}]
    from llmbench.containers.config import CleanupEvidence
    cleanup = CleanupEvidence(attempted=True, verified=False, server_exit="signal=SIGKILL",
                              processes_remaining=("pid:4242",), error="owned server processes remain")
    text = render(config, result_for(config, memory=memory, state="cleanup-uncertain", abort_campaign=True,
                                     failure_reasons=("cleanup unverified",), cleanup=cleanup))
    assert f"watchdog violations: {reason}\n" in text
    assert ("- Cleanup verified: **False**; abort campaign: True; server exit signal=SIGKILL; still present: "
            "pid:4242\n") in text


def test_the_report_reads_the_memory_block_the_native_runner_actually_seals(tmp_path):
    """Contract check against `NativeRunner` itself (its test harness: fake process, telemetry and evaluator), not a
    hand-built block: a key renamed on either side would otherwise turn every figure into `unmeasured` silently."""
    from test_containers_native_runner import Harness
    harness = Harness(tmp_path)
    result = harness.run()
    assert result.state == "completed", result.failure_reasons
    evidence = unified_memory_evidence(result.memory)
    measured = ("server_footprint_mib", "server_rss_mib", "server_footprint_peak_mib", "metal_mib", "metal_budget_mib",
                "host_available_mib", "min_host_available_mib", "swap_used_mib", "swap_growth_mib", "pressure_level",
                "max_pressure_level", "power", "watchdog_violations")
    assert [name for name in measured if evidence[name] is None] == []
    assert evidence["coding_sandbox_status"] is None  # no broker configured: the sandbox was never probed
    text = (harness.output / result.reports["candidate"]).read_text(encoding="utf-8")
    assert ("- Unified memory after load: server footprint 1905 MiB (RSS 952 MiB); Metal buffers 1631 MiB of 5461 "
            "MiB budget; host available 6144 MiB; swap used 6800 MiB (growth 0 MiB); pressure level 1 (normal); "
            "power: battery 70%\n") in text
    assert "report_failed" not in " ".join(result.warnings) and "VRAM after load unmeasured" not in text


def test_write_candidate_reports_succeeds_for_a_metal_candidate(tmp_path):
    """Regression: the candidate report read `config.inference_image.reference`, which a native config does not
    have, so every native run ended with `report_failed: AttributeError`."""
    config = metal_config()
    result = result_for(config, samples=[TOOLS])
    files = write_candidate_reports(config, result, {"samples": [TOOLS]}, None, tmp_path / "run")
    text = (tmp_path / "run" / files["candidate"]).read_text(encoding="utf-8")
    assert "# Native (Metal) candidate" in text and "Unified memory after load" in text


# ---- session report (session_report.py) -----------------------------------------------------------------------


def test_unified_memory_evidence_reads_only_what_was_measured():
    evidence = unified_memory_evidence(MEMORY)
    assert (evidence["server_footprint_mib"], evidence["server_rss_mib"], evidence["metal_mib"],
            evidence["metal_budget_mib"]) == (1905.0, 952.5, 1631.19, 5461.0)
    assert evidence["swap_growth_mib"] == 128.0 and evidence["server_footprint_peak_mib"] == pytest.approx(1905.0)
    assert evidence["power"] == "battery 73%" and evidence["watchdog_violations"] == []
    # Anything that is not an apple-unified block yields nothing: an NVIDIA result has no such block.
    for foreign in (None, {}, {"kind": "nvidia-vram", "after_load": MEMORY["after_load"]}, "apple-unified"):
        assert set(unified_memory_evidence(foreign).values()) == {None}
    # A figure is never parsed out of text or a boolean, and a missing part is None, never zero.
    odd = {"kind": "apple-unified", "after_load": {"server_phys_footprint_mib": "1905", "swap_used_mib": True}}
    odd_evidence = unified_memory_evidence(odd)
    assert odd_evidence["server_footprint_mib"] is None and odd_evidence["swap_used_mib"] is None
    assert odd_evidence["swap_growth_mib"] is None and odd_evidence["watchdog_violations"] is None


@pytest.mark.parametrize("power,expected", [
    ({"power_source": "battery", "battery_percent": 73}, "battery 73%"),
    ({"power_source": "battery", "battery_percent": None}, "battery"),
    ({"power_source": "ac", "battery_percent": 100}, "AC"),
    ({"power_source": None, "power_source_label": "UPS Power"}, "UPS Power"),
    ({"power_source": None, "power_source_label": ""}, None),
    (None, None), ("battery", None)])
def test_power_is_labelled_from_the_reading_and_never_assumed(power, expected):
    assert power_text(power) == expected


def test_pressure_levels_are_named_and_unknown_levels_stay_numbers():
    assert [pressure_text(level) for level in (1, 2, 4, 3, None, True)] == [
        "1 (normal)", "2 (warn)", "4 (critical)", "3", None, None]


def test_columns_are_appended_after_the_nvidia_ones():
    assert COLUMNS[:29][-1] == "artifact_dir" and HEADERS[:29][-1] == "Artifact dir"
    assert COLUMNS[29:] == ("runtime", *UNIFIED_COLUMNS)
    assert HEADERS[29:] == ("Runtime", "Server footprint MiB (unified)", "Metal buffers MiB", "Swap growth MiB",
                            "Power")


def test_nvidia_session_rows_carry_the_runtime_and_dashes_and_no_unified_note(tmp_path, isolated_gpu_lock):
    outcome, runner, clock = run_tune(tmp_path, small_session(tmp_path))
    report = outcome["report"]
    assert {row["runtime"] for row in report["rows"]} == {"nvidia-container"}
    assert all(row[column] is None for row in report["rows"] for column in UNIFIED_COLUMNS)
    assert all(row["unified_memory"] is None for row in report["rows"])
    assert not has_native_rows(report)
    markdown = (tmp_path / "out" / "reports" / "session.md").read_text(encoding="utf-8")
    table = [line for line in markdown.splitlines() if line.startswith("| ") and "baseline" in line]
    assert table and all(line.endswith("| nvidia-container | - | - | - | - |") for line in table)
    assert "Measured llama.cpp container session. Every attempt is listed; failures stay in the table." in markdown
    html_text = (tmp_path / "out" / "reports" / "session.html").read_text(encoding="utf-8")
    for text in (markdown, html_text):
        assert "Apple Silicon" not in text and "native (Metal)" not in text


def _metal_row(tmp_path, memory, *, vram=None) -> dict:
    """The first attempt of a finished fake tune, re-read with its sealed result turned into a Metal one."""
    run_tune(tmp_path, small_session(tmp_path))
    with Store(tmp_path / "out") as store:
        attempt = store.attempts_in_insertion_order()[0]
        path = store.root / "raw" / attempt["id"] / "raw.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["result"].update(runtime="metal-native", memory=memory, vram_used_mib_after_load=vram)
        path.write_text(json.dumps(raw), encoding="utf-8")
        return attempt_row(store, attempt, None)


def test_a_metal_attempt_row_reads_its_result_memory_and_leaves_vram_empty(tmp_path, isolated_gpu_lock):
    # A stray VRAM number in a Metal result is still never printed under `VRAM MiB`: there is no VRAM to measure.
    row = _metal_row(tmp_path, MEMORY, vram=512.0)
    assert row["runtime"] == "metal-native" and row["vram_mib"] is None
    assert (row["server_footprint_mib"], row["metal_mib"], row["swap_growth_mib"], row["power"]) == (
        1905.0, 1631.19, 128.0, "battery 73%")
    assert row["unified_memory"]["pressure_level"] == 2
    report = {"session_id": "s", "campaign_identity": "c", "synthetic": False, "columns": list(COLUMNS),
              "rows": [row], "recommendation": {"preset_note": "n", **{name: {"attempt_id": None, "label": None,
                                                                              "evidence": "none"} for name in (
                  "best_quality", "fastest_eligible", "largest_verified_context")}},
              "comparisons": [], "budget": {"wall_seconds": 1, "charged_seconds": 0, "holdout_reserve_seconds": 0,
                                            "remaining_seconds": 0, "stop_reason": "complete", "started_utc": "t",
                                            "deadline_utc": "t", "candidates_attempted": 1, "holdouts_attempted": 0,
                                            "skipped": []},
              "reproduction": [], "campaign_reports": {}, "candidate_reports": {}}
    markdown = session_markdown(report)
    assert "Measured llama.cpp native (Metal) session." in markdown and UNIFIED_MEMORY_NOTE in markdown
    line = next(line for line in markdown.splitlines() if line.startswith("| baseline"))
    cells = line.strip("| ").split(" | ")
    assert cells[HEADERS.index("VRAM MiB")] == "-"
    assert cells[HEADERS.index("Runtime"):] == ["metal-native", "1905.0", "1631.2", "128.0", "battery 73%"]
    html_text = session_html(report)
    assert "native (Metal) session" in html_text and "Apple Silicon (`metal-native` rows)" in html_text


def test_a_metal_row_whose_sandbox_was_blocked_prints_blocked_not_a_coding_score(tmp_path, isolated_gpu_lock):
    memory = {**MEMORY, "coding_sandbox": {"status": "blocked", "reason": "the Docker daemon is not reachable"}}
    row = _metal_row(tmp_path, memory)
    assert row["coding"] == SESSION_SANDBOX_BLOCKED and row["first_attempt_vs_repaired"] == SESSION_SANDBOX_BLOCKED
    assert "0.000" not in row["coding"]


def test_an_available_sandbox_leaves_the_coding_cell_to_the_evidence(tmp_path, isolated_gpu_lock):
    memory = {**MEMORY, "coding_sandbox": {"status": "available", "reason": None}}
    row = _metal_row(tmp_path, memory)
    assert row["coding"] != SESSION_SANDBOX_BLOCKED and row["first_attempt_vs_repaired"] != SESSION_SANDBOX_BLOCKED


def test_attempt_runtime_prefers_the_result_then_the_pinned_manifest(manifest):
    assert attempt_runtime({"state": "completed"}, manifest) == "nvidia-container"  # key omitted = the default
    assert attempt_runtime({"runtime": "metal-native"}, manifest) == "metal-native"
    assert attempt_runtime({"runtime": "rocm"}, manifest) == "unknown (rocm)"  # never mapped to the default
    metal = manifest.model_copy(update={"backend": manifest.backend.model_copy(
        update={"runtime_revision": "metal:b11011-aa39d7a3e+0123456789ab"})})
    assert attempt_runtime({}, metal) == "metal-native" and attempt_runtime({}, manifest) == "nvidia-container"


# ---- cross-model report (scripts/cross_model_report.py) -------------------------------------------------------


def _session(root: Path, slug: str, *, runtime=None, results=(), blocked=False) -> None:
    """A finished sweep entry: ledger, session config, per-candidate result.json and the benchmark plan."""
    run = root / slug / "run"
    labels = [label for label, _ in results]
    (run / "reports").mkdir(parents=True)
    (run / "session.json").write_text(json.dumps({"session_id": slug, "labels": labels,
                                                  "proposals": [{"changes": {}} for _ in labels],
                                                  "manifests": [f"m-{slug}-{i}" for i in range(len(labels))]}))
    base = {"label": "x"} if runtime is None else {"label": "x", "runtime": runtime}
    (run / "session-config.json").write_text(json.dumps({"base": base}))
    (run / "session-summary.json").write_text(json.dumps({"stop_reason": "complete", "winner": None}))
    analysed = []
    for index, (label, result) in enumerate(results):
        directory = run / "candidates" / f"{index:03d}-{label}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(result))
        analysed.append({"manifest_hash": f"m-{slug}-{index}", "speed": result["speed"],
                         "quality": result["quality"], "eligibility": {"eligible": False, "reasons": []}})
    (run / "reports" / "campaign.json").write_text(json.dumps({"results": analysed}))
    rows = [{"benchmark_id": "bfcl", "status": "selected", "reason": "present"}]
    plan = {"selected": ["bfcl"], "skipped": [], "benchmarks": rows}
    if blocked:
        reason = ("the Docker sandbox is unavailable (docker CLI not found); generated code is never executed on "
                  "the host")
        rows += [{"benchmark_id": suite, "status": "blocked", "reason": reason}
                 for suite in ("evalplus", "aider-polyglot")]
        plan.update(sandbox={"available": False, "reason": "docker CLI not found"},
                    blocked=["evalplus", "aider-polyglot"])
    (run / "benchmark-selection.json").write_text(json.dumps(plan))


def _result(*, vram=None, runtime=None, memory=None, tps=60.0, coding=None):
    result = {"state": "completed", "speed": {"median_native_tps": tps}, "vram_used_mib_after_load": vram,
              "quality": {"tools": {"attempted": 4, "score": 0.75}, "retrieval": {"attempted": 0, "score": None},
                          "coding": coding or {"attempted": 0, "score": None}},
              "settings": [{"control": "cache_type_k", "effective": "f16"}, {"control": "cache_type_v",
                                                                            "effective": "f16"}]}
    if runtime:
        result.update(runtime=runtime, memory=memory if memory is not None else MEMORY)
    return result


def _records(root: Path) -> list[dict]:
    index = {path.name: {"model": None} for path in sorted(root.iterdir()) if path.is_dir()}
    return [cross_model_report.collect_model(slug, meta, root) for slug, meta in index.items()]


def test_an_nvidia_only_sweep_renders_the_original_tables(tmp_path):
    _session(tmp_path, "gpu", results=[("baseline", _result(vram=21042.0)), ("kv-q8", _result(vram=20500.0))])
    records = _records(tmp_path)
    assert records[0]["runtime"] == "nvidia-container" and "blocked" not in records[0]["benchmark_selection"]
    text = cross_model_report.render_markdown(records, "now")
    assert ("| Model | Candidate | ctx | KV | spec | reason | tok/s | VRAM MiB | tools | retrieval | coding | "
            "largest prompt | ctx ok | state |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n") in text
    assert ("| Model | baseline tok/s | best tok/s (settings) | lowest VRAM | tok/s at 64K | VRAM at 262144 | "
            "tools range |\n|---|---|---|---|---|---|---|\n") in text
    assert "| `gpu` | 60.0 | 60.0 (baseline) | 20500 | - | - | 0.750-0.750 |" in text
    assert "| `gpu` | baseline | - | f16/f16 | - | - | 60.0 | 21042 | 0.750 | 0/0 | 0/0 | - | - | completed |" in text
    for native_only in ("runtime", "unified", "Unified", "blocked", "Runtime", "Metal"):
        assert native_only not in text, native_only


def test_a_mixed_sweep_labels_memory_by_kind_and_never_minimises_across_kinds(tmp_path):
    _session(tmp_path, "gpu", results=[("baseline", _result(vram=21042.0))])
    # A Metal result carrying a stray VRAM figure must still never be read as VRAM, nor enter a VRAM minimum.
    metal = [("baseline", _result(runtime="metal-native", vram=100.0, tps=18.0)),
             ("kv-q8", _result(runtime="metal-native", tps=17.0,
                               memory={**MEMORY, "after_load": {**MEMORY["after_load"],
                                                                "server_phys_footprint_mib": 1700.0}}))]
    _session(tmp_path, "mac", runtime="metal-native", results=metal, blocked=True)
    records = _records(tmp_path)
    assert [record["runtime"] for record in records] == ["nvidia-container", "metal-native"]
    assert records[1]["candidates"][0]["vram_mib"] is None and records[0]["candidates"][0]["vram_mib"] == 21042.0
    text = cross_model_report.render_markdown(records, "now")
    assert "These sessions ran on different runtimes (metal-native, nvidia-container)" in text
    rows = {line.split(" | ")[1]: line for line in text.splitlines() if line.startswith("| `mac` | ")}
    baseline = rows["baseline"]
    assert "| 18.0 | n/a (unified) | 0.750 | 0/0 | blocked |" in baseline and "| 100 |" not in baseline
    assert baseline.endswith("| completed | metal-native | 1905 | 1631 | 128 | battery 73% |")
    gpu = next(line for line in text.splitlines() if line.startswith("| `gpu` | baseline"))
    assert gpu.endswith("| completed | nvidia-container | - | - | - | - |") and "| 21042 |" in gpu
    comparison = next(line for line in text.splitlines() if line.startswith("| `mac` | 18.0"))
    assert comparison == "| `mac` | 18.0 | 18.0 (baseline) | n/a (unified) | - | n/a (unified) | 0.750-0.750 | 1700 |"
    gpu_comparison = next(line for line in text.splitlines() if line.startswith("| `gpu` | 60.0"))
    assert gpu_comparison.endswith("| 21042 | - | - | 0.750-0.750 | - |")
    assert "- Blocked: `evalplus` - the Docker sandbox is unavailable (docker CLI not found)" in text
    assert "**Coding is blocked** for `mac`" in text and "**Unified memory is not VRAM.**" in text
    assert records[1]["benchmark_selection"]["blocked"][0]["benchmark_id"] == "evalplus"


def test_a_candidate_whose_sandbox_was_blocked_at_run_time_shows_blocked_not_its_backfilled_zero(tmp_path):
    memory = {**MEMORY, "coding_sandbox": {"status": "blocked", "reason": "the Docker daemon is not reachable"}}
    _session(tmp_path, "mac", runtime="metal-native", results=[(
        "baseline", _result(runtime="metal-native", memory=memory, coding={"attempted": 5, "score": 0.0}))])
    records = _records(tmp_path)
    assert records[0]["candidates"][0]["coding_blocked"] == "the Docker daemon is not reachable"
    line = next(line for line in cross_model_report.render_markdown(records, "now").splitlines()
                if line.startswith("| `mac` | baseline"))
    assert "| blocked |" in line and "0.000" not in line.split(" | ")[10]


# ---- coding report (scripts/coding_report.py) -----------------------------------------------------------------


def _coding_candidate(label, passed, *, quality=42.0, blocked=None, n=None):
    ids = [f"Mbpp/{i}" for i in range(len(passed) if n is None else n)]
    return {"label": label, "state": "completed", "items": {t: {"task_id": t, "passed": p} for t, p in
                                                            zip(ids, passed)},
            "finish": {"stop": len(passed)}, "tokens": [100] * len(passed), "quality_seconds": quality,
            "tps": 60.0, "vram": None, "environment_errors": 0, "blocked": blocked}


def test_the_nvidia_coding_table_is_unchanged():
    text = coding_report.render({"gpu": [_coding_candidate("baseline", [True, True, False])]}, "evalplus")
    assert "| `gpu` | baseline | **2/3 = 66.7%** | 21%-94% | 0 | 0 | 100 | 42 s |" in text
    assert "blocked" not in text


def test_an_empty_suite_or_an_untimed_stage_never_divides_by_zero():
    text = coding_report.render({"m": [_coding_candidate("baseline", []),
                                       _coding_candidate("reasoning-on", [True], quality=None)]}, "evalplus")
    assert "| `m` | baseline | 0/0 (no items) | - | - | - | - | 42 s |" in text
    reasoning = next(line for line in text.splitlines() if line.startswith("| `m` | reasoning-on"))
    assert reasoning.startswith("| `m` | reasoning-on | **1/1 = 100.0%** |") and reasoning.endswith("| 100 | - |")
    assert "reasoning off vs on" not in text  # nothing to pair against an empty suite
    polyglot = coding_report.render({"m": [_coding_candidate("baseline", [], quality=None)]}, "aider-polyglot")
    assert "| `m` | baseline | 0/0 (no items) |" + " - |" * 9 in polyglot


def test_a_blocked_suite_is_printed_blocked_and_kept_out_of_paired_comparisons():
    reason = "the Docker sandbox is unavailable (docker CLI not found); generated code is never executed on the host"
    models = {"gpu": [_coding_candidate("baseline", [True, False])],
              "mac": [_coding_candidate("baseline", [])], "arm": []}
    text = coding_report.render(models, "evalplus", {"mac": reason, "arm": reason})
    assert f"| `mac` | baseline | blocked: {reason} |" + " - |" * 5 in text
    assert f"| `arm` | *(not run)* | blocked: {reason} |" + " - |" * 5 in text
    assert "vs" not in text.split("## Paired comparisons")[1].split("A p-value")[0]
    assert "is not a pass rate of zero" in text


def test_a_sandbox_blocked_at_run_time_is_read_from_the_evaluation_and_never_scored(tmp_path):
    root = tmp_path / "sweep"
    (root / "index.json").parent.mkdir()
    (root / "index.json").write_text(json.dumps({"gpu": {}, "mac": {}}))
    why = "sandbox_unavailable: docker CLI not found"
    for slug, rows, records in (
            ("gpu", [{"task_id": "Mbpp/1", "suite": "evalplus", "passed": True, "status": "completed"}], []),
            ("mac", [{"task_id": "Mbpp/1", "suite": "evalplus", "passed": False, "status": "environment_error",
                      "reason": why}],
             [{"benchmark_id": "evalplus", "status": "environment_error", "reason": why}])):
        directory = root / slug / "run" / "candidates" / "000-baseline"
        (directory / "evaluator").mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"state": "completed", "stages": [
            {"name": "quality", "elapsed_seconds": 30.0}]}))
        (directory / "evaluator" / "evaluation.json").write_text(json.dumps({"samples": rows, "benchmarks": records}))
    assert coding_report.main(["--root", str(root)]) == 0
    text = (root / "coding-report-evalplus.md").read_text(encoding="utf-8")
    assert f"| `mac` | baseline | blocked: {why} |" in text and "0/1" not in text
    assert "| `gpu` | baseline | **1/1 = 100.0%** |" in text
    assert "`gpu` vs `mac`" not in text  # a blocked baseline is never paired against a real one
