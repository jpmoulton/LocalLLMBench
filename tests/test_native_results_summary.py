"""scripts/native_results_summary.py reads finished run directories and never turns a harness failure or an
output cap into a model score."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import native_results_summary as summary  # noqa: E402

MIB = 1024 * 1024


def write_candidate(root: Path, name: str, *, samples, speed=None, memory=None, runtime="metal-native"):
    run = root / "run" / "candidates" / name
    run.mkdir(parents=True)
    config = {"label": name, "runtime": runtime, "assets": [{"model_name": "Tiny", "sha256": "a" * 64}],
              "engine": {"ctx_size": 4864, "cache_type_k": "f16", "cache_type_v": "f16", "reasoning": "on"},
              "native_server": {"build_info": "b11011-aa39d7a3e"}, "native_limits": {"memory_reserve_mib": 0}}
    (run / "config.json").write_text(json.dumps(config))
    (run / "result.json").write_text(json.dumps({"state": "completed", "speed": speed or {}, "memory": memory or {},
                                                 "effective_settings_verified": True,
                                                 "actual_context_verified": False, "cleanup": {"verified": True}}))
    (run / "evaluation.json").write_text(json.dumps({"samples": samples, "native_timings": [
        {"prompt_n": 4000, "prompt_ms": 8000.0}]}))
    return run


def test_harness_errors_are_never_scored_and_a_cut_off_row_is_counted_once(tmp_path):
    write_candidate(tmp_path, "000-baseline", samples=[
        {"suite": "ruler", "task_id": "ruler/a", "status": "environment_error", "score": 0.0},
        {"suite": "ruler", "task_id": "ruler/b", "status": "completed", "score": 1.0},
        {"suite": "ruler", "task_id": "ruler/c", "status": "invalid_output", "finish_reason": "length", "score": 0},
        {"suite": "bfcl", "task_id": "bfcl/x", "status": "completed", "score": 1.0, "finish_reason": "tool_calls"},
        {"suite": "bfcl", "task_id": "bfcl/y", "status": "invalid_output", "score": 0.0},
        {"suite": "niah", "task_id": "niah/m", "status": "completed", "score": 0.0, "lenient_outcome_correct": True},
    ])
    [row] = summary.collect([tmp_path])
    ruler, bfcl, niah = row["suites"]["ruler"], row["suites"]["bfcl"], row["suites"]["niah"]
    assert ruler == {"items": 3, "scored": 2, "passed": 1.0, "score": 0.5, "harness_errors": 1,
                     "failed_not_cut_off": 0, "truncated": 1, "lenient_correct": 0}
    assert bfcl["score"] == 0.5 and bfcl["failed_not_cut_off"] == 1 and bfcl["truncated"] == 0
    assert niah["score"] == 0.0 and niah["lenient_correct"] == 1  # strict and lenient both kept
    table = summary.markdown([row])
    assert "1/2 (1 harness err, 1 cut off)" in table and "1/2 (1 invalid)" in table and "0/1 (1/1)" in table
    assert row["prefill_tps_median"] == 500.0


def test_a_candidate_without_speed_evidence_has_no_tok_s_and_metal_rows_carry_no_vram(tmp_path):
    memory = {"kind": "apple-unified", "after_load": {"server_phys_footprint_mib": 1901.2},
              "during_evaluation": {"peak_phys_footprint_bytes": 1929 * MIB, "swap_growth_bytes": 889 * MIB,
                                    "max_pressure_level": 2},
              "server_log": {"metal_resident_mib": 1631.19, "offloaded_layers": 29, "total_layers": 29}}
    write_candidate(tmp_path, "002-kv-q4_0-q4_0", samples=[],
                    speed={"median_native_tps": None, "reasons": ["repetition_0:incomplete"]}, memory=memory)
    [row] = summary.collect([tmp_path])
    assert row["tps_median"] is None and row["speed_reasons"] == ["repetition_0:incomplete"]
    assert row["vram_mib"] is None and row["server_footprint_peak_mib"] == 1929 and row["offloaded"] == "29/29"
    line = summary.markdown([row]).splitlines()[-1]
    assert "| - |" in line and "1901 / 1929" in line and "| 889 |" in line


def test_the_command_refuses_roots_without_candidates(tmp_path, capsys):
    assert summary.main([str(tmp_path)]) == 2
    assert "no candidates" in capsys.readouterr().err


def test_cut_off_responses_are_counted_per_suite_once_even_when_two_files_hold_them(tmp_path):
    run = write_candidate(tmp_path, "000-baseline", samples=[])
    def response(path, finish):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"choices": [{"finish_reason": finish, "message": {"content": "x"}}]}))
    aider = run / "evaluator" / "benchmarks" / "aider-polyglot"
    response(aider / "aider-polyglot" / "python" / "bowling" / "response-1.json", "length")  # adapter copies
    response(aider / "aider-polyglot" / "python" / "bowling" / "response-2.json", "stop")
    response(aider / "request-0.json", "length")  # the transport's copies of the same two completions
    response(aider / "request-1.json", "stop")
    response(run / "evaluator" / "benchmarks" / "bfcl" / "request-0.json", "length")  # transport-only suite
    response(run / "evaluator" / "coding" / "python-chunks" / "response-1.json", "stop")
    (run / "evaluator" / "benchmarks" / "bfcl" / "metrics.json").write_text(json.dumps({"items": 1}))
    [row] = summary.collect([tmp_path])
    assert row["responses_cut_off"] == {"aider-polyglot": {"cut_off": 1, "responses": 2},
                                        "bfcl": {"cut_off": 1, "responses": 1},
                                        "coding": {"cut_off": 0, "responses": 1}}
