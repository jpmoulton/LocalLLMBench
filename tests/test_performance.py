"""CPU-only coverage of native generation evidence and the preparation boundary."""

import json
import sys
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.performance import collect_generation_metrics
from benchmarks.run import run_benchmark


def model_event(*, latency=2.0, prompt_tokens=20, prompt_ms=100.0,
                generated_tokens=100, decode_ms=1000.0, **fields):
    event = {
        "event": "model",
        "call": {
            "time": latency,
            "response": {"timings": {
                "cache_n": 5, "prompt_n": prompt_tokens, "prompt_ms": prompt_ms,
                "predicted_n": generated_tokens, "predicted_ms": decode_ms,
            }},
        },
        "output": SimpleNamespace(
            usage=SimpleNamespace(input_tokens=25, output_tokens=generated_tokens),
            choices=[SimpleNamespace(stop_reason="stop")], time=999,
        ),
    }
    event.update(fields)
    return SimpleNamespace(**event)


def sample(events, **fields):
    data = {"id": "arbitrary/task:id", "epoch": 2, "events": events,
            "scores": {"quality": SimpleNamespace(value=0.75)}, "output": None}
    data.update(fields)
    return SimpleNamespace(**data)


def extract(*samples):
    return collect_generation_metrics([SimpleNamespace(samples=list(samples))])


class GenerationMetricsTests(unittest.TestCase):
    def test_weighted_timings_and_multiple_calls_do_not_double_count_sample_output(self):
        result = extract(sample([
            model_event(),
            model_event(latency=6, generated_tokens=300, decode_ms=3000,
                        prompt_tokens=60, prompt_ms=300),
        ], output=SimpleNamespace(time=777, usage=SimpleNamespace(output_tokens=9000))))
        metrics = result["performance"]
        self.assertEqual(metrics["model_calls"], 2)
        self.assertEqual(metrics["latency_calls"], 2)
        self.assertEqual(metrics["decode_calls"], 2)
        self.assertEqual(metrics["prompt_calls"], 2)
        self.assertEqual(metrics["request_latency_seconds_mean"], 4)
        self.assertEqual(metrics["request_latency_seconds_total"], 8)
        self.assertEqual(metrics["generated_tokens"], 400)
        self.assertEqual(metrics["decode_seconds"], 4)
        self.assertEqual(metrics["decode_tokens_per_second"], 100)
        self.assertEqual(metrics["decode_ms_per_token"], 10)
        self.assertEqual(metrics["prompt_tokens"], 80)
        self.assertAlmostEqual(metrics["prompt_seconds"], 0.4)
        self.assertAlmostEqual(metrics["prompt_tokens_per_second"], 200)
        self.assertEqual(metrics["output_fallback_calls"], 0)
        rows = result["generation_samples"]
        self.assertEqual([row["call_index"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["id"], "arbitrary/task:id")
        self.assertEqual(rows[0]["epoch"], 2)
        self.assertEqual(rows[0]["input_tokens"], 25)
        self.assertEqual(rows[0]["prompt_cached_tokens"], 5)
        self.assertEqual(rows[0]["scores"], {"quality": 0.75})

    def test_throughput_is_ratio_of_sums_not_mean_of_request_rates(self):
        result = extract(sample([
            model_event(generated_tokens=100, decode_ms=1000),
            model_event(generated_tokens=100, decode_ms=3000),
        ]))
        self.assertEqual(result["performance"]["decode_tokens_per_second"], 50)
        self.assertEqual(result["performance"]["decode_ms_per_token"], 20)

    def test_partial_measurement_coverage_does_not_mix_unpaired_tokens_and_time(self):
        result = extract(sample([
            model_event(),
            model_event(latency=None, generated_tokens=800, decode_ms=None,
                        prompt_tokens=None, prompt_ms=900),
            model_event(latency=None, generated_tokens=None, decode_ms=8000,
                        prompt_tokens=900, prompt_ms=None),
        ]))
        metrics = result["performance"]
        self.assertEqual(metrics["model_calls"], 3)
        self.assertEqual(metrics["latency_calls"], 1)
        self.assertEqual(metrics["decode_calls"], 1)
        self.assertEqual(metrics["prompt_calls"], 1)
        self.assertEqual(metrics["generated_tokens"], 100)
        self.assertEqual(metrics["decode_seconds"], 1)
        self.assertEqual(metrics["prompt_tokens"], 20)
        self.assertEqual(result["generation_samples"][1]["generated_tokens"], 800)
        self.assertIsNone(result["generation_samples"][1]["request_latency_seconds"])

    def test_invalid_timing_is_unavailable_and_remains_json_safe(self):
        for invalid in (None, -1, float("nan"), float("inf"), True, "100"):
            with self.subTest(invalid=invalid):
                result = extract(sample([model_event(
                    latency=invalid, prompt_ms=invalid, decode_ms=invalid,
                )]))
                metrics = result["performance"]
                self.assertEqual(metrics["model_calls"], 1)
                self.assertEqual(metrics["latency_calls"], 0)
                self.assertEqual(metrics["decode_calls"], 0)
                self.assertEqual(metrics["prompt_calls"], 0)
                self.assertIsNone(metrics["decode_tokens_per_second"])
                self.assertIsNone(metrics["request_latency_seconds_mean"])
                self.assertIsNone(metrics["generated_tokens"])
                json.dumps(result, allow_nan=False)

    def test_zero_duration_is_not_throughput_but_zero_tokens_are_measured(self):
        result = extract(sample([model_event(
            latency=0, generated_tokens=0, decode_ms=10, prompt_tokens=0, prompt_ms=0,
        )]))
        metrics = result["performance"]
        self.assertEqual(metrics["latency_calls"], 1)
        self.assertEqual(metrics["request_latency_seconds_total"], 0)
        self.assertEqual(metrics["decode_calls"], 1)
        self.assertEqual(metrics["decode_tokens_per_second"], 0)
        self.assertIsNone(metrics["decode_ms_per_token"])
        self.assertEqual(metrics["prompt_calls"], 0)
        self.assertIsNone(metrics["prompt_tokens_per_second"])
        self.assertEqual(result["generation_samples"][0]["prompt_seconds"], 0)

    def test_empty_and_unstarted_logs_do_not_fabricate_calls(self):
        result = collect_generation_metrics([
            SimpleNamespace(samples=None), {"samples": [
                {"id": 1, "events": [], "output": {"choices": []}},
            ]},
        ])
        self.assertEqual(result["generation_samples"], [])
        self.assertEqual(result["performance"]["model_calls"], 0)
        self.assertIsNone(result["performance"]["request_latency_seconds_total"])
        self.assertIsNone(result["performance"]["decode_seconds"])

    def test_output_fallback_has_latency_and_usage_but_no_invented_server_timings(self):
        result = extract(sample([
            SimpleNamespace(event="model", call=None),
            SimpleNamespace(event="score", call={"time": 111}),
        ], output={"time": 4, "usage": {"input_tokens": 12, "output_tokens": 80},
                   "choices": [{"stop_reason": "max_tokens"}]}))
        metrics = result["performance"]
        self.assertEqual(metrics["model_calls"], 1)
        self.assertEqual(metrics["output_fallback_calls"], 1)
        self.assertEqual(metrics["model_events_without_call"], 1)
        self.assertEqual(metrics["request_latency_seconds_mean"], 4)
        self.assertEqual(metrics["decode_calls"], 0)
        row = result["generation_samples"][0]
        self.assertEqual(row["output_tokens"], 80)
        self.assertEqual(row["finish_reason"], "max_tokens")
        self.assertIsNone(row["generated_tokens"])
        self.assertIsNone(row["decode_tokens_per_second"])

    def test_error_and_truncated_calls_are_preserved_and_measured(self):
        failure = SimpleNamespace(message="request failed", details={"code": 500})
        truncated = model_event(error=failure)
        truncated.output.choices[0].stop_reason = "max_tokens"
        result = extract(sample([truncated, SimpleNamespace(event="model", call=None,
                                                          error={"message": "connection lost"})]))
        self.assertEqual(result["performance"]["model_calls"], 2)
        self.assertEqual(result["performance"]["decode_calls"], 1)
        row, missing = result["generation_samples"]
        self.assertEqual(row["finish_reason"], "max_tokens")
        self.assertEqual(row["error"]["details"]["code"], 500)
        self.assertEqual(missing["error"], {"message": "connection lost"})
        self.assertIsNone(missing["decode_seconds"])
        json.dumps(result, allow_nan=False)

    def test_mapping_logs_use_raw_response_usage_and_sample_error(self):
        result = collect_generation_metrics([{"samples": [{
            "id": 42, "epoch": 1, "scores": {"judge": {"value": "C"}},
            "error": {"message": "scoring failed"},
            "events": [{"event": "model", "call": {"time": 2, "response": {
                "usage": {"prompt_tokens": 15, "completion_tokens": 30},
                "choices": [{"finish_reason": "length"}],
                "timings": {"predicted_n": 30, "predicted_ms": 1500},
            }}}],
        }]}])
        row = result["generation_samples"][0]
        self.assertEqual(row["id"], "42")
        self.assertEqual(row["input_tokens"], 15)
        self.assertEqual(row["output_tokens"], 30)
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["error"], {"message": "scoring failed"})
        self.assertEqual(row["scores"], {"judge": "C"})
        self.assertEqual(result["performance"]["decode_tokens_per_second"], 20)


class ExpectedManifestTests(unittest.TestCase):
    def test_changed_or_missing_identity_stops_before_runtime_and_preserves_evidence(self):
        actual = {"selected_ids": ["one", "two"], "fingerprint": "new"}
        definition = SimpleNamespace(
            prepare=lambda options: (["payload"], actual),
            SCORE_METRICS={"accuracy": {"direction": "higher"}},
        )
        for expected in ({}, {"example": {**actual, "fingerprint": "old"}},
                         {"example": {**actual, "selected_ids": ["two", "one"]}}):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "example"
                with patch("benchmarks.run.get_definition", return_value=definition), \
                        patch("benchmarks.run.get_runtime", side_effect=AssertionError(
                            "Identity failure must not load runtime or generate")) as runtime:
                    result = run_benchmark("example", {}, None, None, {}, output, expected)
                runtime.assert_not_called()
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["run_error"]["type"], "ValueError")
                self.assertIn("expected plan identity", result["run_error"]["message"])
                self.assertEqual(result["performance"]["model_calls"], 0)
                self.assertEqual(result["score_metrics"], definition.SCORE_METRICS)
                self.assertEqual(json.loads((output / "manifest.json").read_text()), actual)
                self.assertEqual(json.loads((output / "summary.json").read_text()), result)

    def test_matching_identity_keeps_metrics_when_benchmark_summary_fails(self):
        manifest = {"selected_ids": ["one"], "fingerprint": "stable"}
        native_logs = [SimpleNamespace(samples=[sample([model_event()])])]
        definition = SimpleNamespace(prepare=lambda options: (["payload"], manifest))

        def broken_summary(logs, provenance, error):
            raise ValueError("scorer summary unavailable")

        runtime = SimpleNamespace(
            make_task=lambda payload, provenance, options, generation: "task",
            summarize=broken_summary,
        )
        inspect_module = ModuleType("inspect_ai")
        inspect_module.eval = lambda *args, **kwargs: native_logs
        with tempfile.TemporaryDirectory() as directory:
            with patch("benchmarks.run.get_definition", return_value=definition), \
                    patch("benchmarks.run.get_runtime", return_value=runtime), \
                    patch.dict(sys.modules, {"inspect_ai": inspect_module}):
                result = run_benchmark(
                    "example", {}, None, None, {"concurrency": 1},
                    Path(directory) / "example", {"example": manifest.copy()},
                )
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["summary_error"]["message"], "scorer summary unavailable")
        self.assertEqual(result["performance"]["decode_tokens_per_second"], 100)
        self.assertEqual(len(result["generation_samples"]), 1)
        self.assertEqual(result["score_metrics"], {})


if __name__ == "__main__":
    unittest.main()
