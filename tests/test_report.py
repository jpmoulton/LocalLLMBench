"""CPU-only report exports from persisted benchmark evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook

from orchestrator.report import write_report


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.plan = {
            "name": "=experiment", "source": "cli", "output_dir": str(self.output),
            "benchmarks": ["alpha", "beta"],
            "generation": {"max_tokens": 32}, "execution": {"concurrency": 2},
            "manifests": {"alpha": {"selection_digest": "abc"}},
            "trials": [
                {"id": f"{number:04d}", "model": f"model-{number}.gguf",
                 "group": "@group", "parameters": {"--parallel": 2},
                 "server_args": ["--parallel", "2"]}
                for number in range(1, 6)
            ],
        }
        self.run = {
            "run_id": "run-1", "name": "=experiment", "status": "running",
            "phase": "benchmark", "started_at": "2026-11-03T12:00:00Z",
            "finished_at": None, "elapsed_seconds": 18.5, "current_trial": "0003",
            "counts": {"completed": 1, "incomplete": 1, "running": 1, "pending": 2},
            "errors": [],
            "trials": [
                {"id": "0001", "status": "completed", "duration_seconds": 10.5,
                 "durations": {"load": 1.0, "probe": 0.5, "benchmark": 8.0, "cleanup": 1.0}},
                {"id": "0002", "status": "incomplete", "error": {"message": "=failure"}},
                {"id": "0003", "status": "running"},
            ],
        }

    def save_summary(self, trial_id: str, benchmark: str, summary: dict) -> Path:
        path = self.output / "trials" / trial_id / "benchmarks" / benchmark / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def workbook(self):
        workbook = load_workbook(self.output / "results.xlsx")
        self.addCleanup(workbook.close)
        return workbook

    @staticmethod
    def rows(sheet) -> list[dict]:
        values = list(sheet.values)
        return [dict(zip(values[0], row)) for row in values[1:]]

    def test_declared_scores_rank_per_benchmark_with_ties_and_both_directions(self) -> None:
        metadata = {
            "accuracy": {"label": "Accuracy", "direction": "higher", "unit": "fraction"},
            "loss": {"label": "Loss", "direction": "lower", "unit": "points"},
        }
        for trial_id, accuracy, loss in (("0001", 0.75, 2.0), ("0002", 0.75, 1.0)):
            self.save_summary(trial_id, "alpha", {
                "status": "completed", "accuracy": accuracy, "loss": loss,
                "undeclared_score": 999, "score_metrics": metadata,
                "performance": {"decode_tokens_per_second": 100 if trial_id == "0001" else 200},
            })
        self.save_summary("0003", "alpha", {
            "status": "incomplete", "accuracy": 1.0, "loss": 0.0, "score_metrics": metadata,
        })
        self.save_summary("0001", "beta", {
            "status": "completed", "accuracy": 1.0,
        })
        report = write_report(self.output, self.plan, self.run)
        leaders = report["benchmark_leaders"]
        self.assertEqual(leaders["alpha"]["accuracy"]["trial_ids"], ["0001", "0002"])
        self.assertEqual(leaders["alpha"]["accuracy"]["value"], 0.75)
        self.assertEqual(leaders["alpha"]["loss"]["trial_ids"], ["0002"])
        self.assertEqual(leaders["alpha"]["loss"]["value"], 1.0)
        self.assertEqual(set(leaders["alpha"]), {"accuracy", "loss"})
        self.assertEqual(leaders["beta"], {})
        self.assertEqual(json.loads((self.output / "report.json").read_text(encoding="utf-8")), report)
        workbook = self.workbook()
        self.assertEqual(workbook.sheetnames, ["Run", "Trials", "Benchmark results", "Samples"])
        results = self.rows(workbook["Benchmark results"])
        self.assertEqual(results[0]["performance.decode_tokens_per_second"], 100)
        self.assertEqual(results[2]["performance.decode_tokens_per_second"], 200)
        leader_rows = list(workbook["Run"].values)
        accuracy_row = next(row for row in leader_rows if row[0:2] == ("alpha", "accuracy"))
        self.assertEqual(accuracy_row[4], 0.75)
        self.assertEqual(json.loads(accuracy_row[5]), ["0001", "0002"])

    def test_all_planned_trials_and_missing_results_remain_visible(self) -> None:
        report = write_report(self.output, self.plan, self.run)
        self.assertEqual(len(report["trials"]), 5)
        self.assertEqual(len(report["benchmark_results"]), 10)
        self.assertEqual(report["trials"][-1]["status"], "pending")
        self.assertEqual(report["benchmark_results"][0]["status"], "incomplete")
        self.assertEqual(report["benchmark_results"][-1]["status"], "pending")
        self.assertEqual(report["benchmark_leaders"], {"alpha": {}, "beta": {}})
        self.assertEqual(report["report_errors"], [])
        workbook = self.workbook()
        trials = self.rows(workbook["Trials"])
        self.assertEqual(trials[0]["duration.benchmark_seconds"], 8.0)
        self.assertEqual(trials[-1]["status"], "pending")
        self.assertIsNone(trials[-1]["duration_seconds"])
        self.assertEqual(json.loads(trials[1]["error"]), {"message": "=failure"})
        for row in self.rows(workbook["Benchmark results"]):
            self.assertIsNone(row["performance.generated_tokens"])
            self.assertIsNone(row["performance.model_calls"])

    def test_coverage_samples_and_arbitrary_outcomes_preserve_provenance(self) -> None:
        outcome = {
            "id": "custom/task-A", "inspect_id": "native-id", "status": "truncation",
            "error": {"message": "limit reached"}, "scores": {"grader": 0.2},
            "custom_outcome": {"checks": [True, False]},
        }
        generation = {
            "id": "native-id", "epoch": 1, "call_index": 0,
            "input_tokens": 10, "output_tokens": 8, "request_latency_seconds": 1.5,
            "prompt_tokens": None, "prompt_seconds": None, "generated_tokens": 8,
            "decode_seconds": 0.5, "decode_tokens_per_second": 16,
            "finish_reason": "length", "error": {"message": "limit reached"},
            "scores": {"grader": 0.2}, "timing_source": "model_event",
        }
        summary = {
            "status": "incomplete", "denominator": 3, "scored": 1,
            "started_at": "2026-11-03T12:00:00+00:00",
            "finished_at": "2026-11-03T12:00:02.500000+00:00",
            "counts": {"pass": 1, "truncation": 1, "missing": 1},
            "inspect_logs": ["/output/alpha/inspect/native.eval"],
            "performance": {
                "model_calls": 3, "latency_calls": 2, "decode_calls": 1, "prompt_calls": 0,
                "generated_tokens": 8, "decode_seconds": 0.5, "decode_tokens_per_second": 16,
                "decode_ms_per_token": 62.5, "request_latency_seconds_mean": 1.5,
            },
            "samples": [outcome], "generation_samples": [generation],
        }
        self.save_summary("0001", "alpha", summary)
        report = write_report(self.output, self.plan, self.run)
        self.assertEqual(report["benchmark_results"][0]["summary"], summary)
        self.assertEqual(report["samples"][0]["sample"], outcome)
        self.assertEqual(report["samples"][1]["sample"], generation)
        workbook = self.workbook()
        result = self.rows(workbook["Benchmark results"])[0]
        self.assertEqual(result["benchmark_duration_seconds"], 2.5)
        self.assertEqual(result["performance.model_calls"], 3)
        self.assertEqual(result["performance.decode_calls"], 1)
        self.assertEqual(result["performance.prompt_calls"], 0)
        self.assertIsNone(result["performance.prompt_tokens"])
        self.assertEqual(result["result.scored"], 1)
        self.assertEqual(result["result.denominator"], 3)
        self.assertEqual(json.loads(result["expected_manifest"]), {"selection_digest": "abc"})
        self.assertEqual(result["request_concurrency"], 2)
        self.assertEqual(result["parallel_slots"], 2)
        samples = self.rows(workbook["Samples"])
        self.assertEqual([row["row_type"] for row in samples], ["outcome", "generation"])
        self.assertEqual([row["sample.id"] for row in samples], ["custom/task-A", "native-id"])
        self.assertEqual(samples[1]["sample.generated_tokens"], 8)
        self.assertEqual(samples[1]["sample.decode_seconds"], 0.5)
        self.assertEqual(json.loads(samples[0]["sample.custom_outcome"]), {"checks": [True, False]})
        self.assertEqual(json.loads(samples[1]["inspect_logs"]), summary["inspect_logs"])
        self.assertEqual(samples[1]["summary_path"], "trials/0001/benchmarks/alpha/summary.json")
        sheet = workbook["Samples"]
        headers = [cell.value for cell in sheet[1]]
        link = sheet.cell(3, headers.index("inspect_directory") + 1).hyperlink
        self.assertIsNotNone(link)
        self.assertEqual(link.target, "trials/0001/benchmarks/alpha/inspect")

    def test_external_strings_are_literal_and_numeric_values_remain_numeric(self) -> None:
        self.save_summary("0001", "alpha", {
            "status": "completed", "score_metrics": {
                "score": {"label": "=HYPERLINK(\"bad\")", "direction": "higher"},
            }, "score": 0.5,
            "samples": [{"id": "=1+1", "text": "+cmd", "other": "-cmd", "at": "@cmd",
                         "error": {"message": "=1+1"}, "control": "text\x00end"}],
        })
        report = write_report(self.output, self.plan, self.run)
        self.assertEqual(report["samples"][0]["sample"]["control"], "text\x00end")
        workbook = self.workbook()
        for sheet in workbook:
            for row in sheet:
                for cell in row:
                    self.assertNotEqual(cell.data_type, "f", (sheet.title, cell.coordinate))
        samples = self.rows(workbook["Samples"])
        self.assertEqual(samples[0]["sample.id"], "=1+1")
        self.assertEqual(samples[0]["sample.text"], "+cmd")
        self.assertEqual(samples[0]["sample.other"], "-cmd")
        self.assertEqual(samples[0]["sample.at"], "@cmd")
        self.assertEqual(samples[0]["sample.control"], "text\\x00end")
        self.assertEqual(self.rows(workbook["Benchmark results"])[0]["result.score"], 0.5)
        self.assertEqual(self.rows(workbook["Trials"])[0]["id"], "0001")

    def test_missing_boolean_nonfinite_and_undeclared_scores_never_win(self) -> None:
        metadata = {"score": {"direction": "higher"}}
        for trial_id, score in (("0001", float("nan")), ("0002", float("inf")),
                                ("0003", True), ("0004", "0.9"), ("0005", None)):
            self.save_summary(trial_id, "alpha", {
                "status": "completed", "score": score, "score_metrics": metadata,
            })
        report = write_report(self.output, self.plan, self.run)
        self.assertEqual(report["benchmark_leaders"]["alpha"], {})
        self.assertIsNone(report["benchmark_results"][0]["summary"]["score"])
        parsed = json.loads((self.output / "report.json").read_text(encoding="utf-8"),
                            parse_constant=lambda value: self.fail(f"Nonfinite JSON: {value}"))
        self.assertEqual(parsed, report)

    def test_conflicting_direction_or_units_do_not_define_a_leader(self) -> None:
        for definitions in (
            ({"direction": "higher"}, {"direction": "lower"}),
            ({"direction": "higher", "unit": "fraction"}, {"direction": "higher", "unit": "percent"}),
        ):
            with self.subTest(definitions=definitions):
                for trial_id, definition in zip(("0001", "0002"), definitions):
                    self.save_summary(trial_id, "alpha", {
                        "status": "completed", "score": 0.5, "score_metrics": {"score": definition},
                    })
                report = write_report(self.output, self.plan, self.run)
                self.assertEqual(report["benchmark_leaders"]["alpha"], {})

    def test_malformed_summary_is_visible_without_hiding_other_evaluations(self) -> None:
        broken = self.save_summary("0001", "alpha", {})
        broken.write_text("{unfinished", encoding="utf-8")
        self.save_summary("0002", "alpha", {
            "status": "completed", "score": 0,
            "score_metrics": {"score": {"direction": "higher"}},
        })
        report = write_report(self.output, self.plan, self.run)
        self.assertEqual(report["benchmark_results"][0]["status"], "incomplete")
        self.assertEqual(report["report_errors"][0]["path"], "trials/0001/benchmarks/alpha/summary.json")
        self.assertEqual(report["benchmark_leaders"]["alpha"]["score"]["value"], 0)
        self.assertEqual(report["benchmark_leaders"]["alpha"]["score"]["trial_ids"], ["0002"])

    def test_failed_workbook_save_preserves_previous_file_and_new_json(self) -> None:
        write_report(self.output, self.plan, self.run)
        workbook_bytes = (self.output / "results.xlsx").read_bytes()
        original_names = {path.name for path in self.output.iterdir()}
        self.run["status"] = "cancelled"

        def fail_save(workbook, filename):
            self.assertEqual(Path(filename).suffix, ".xlsx")
            Path(filename).write_bytes(b"incomplete ZIP")
            raise OSError("disk full")

        with patch("orchestrator.report.Workbook.save", new=fail_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                write_report(self.output, self.plan, self.run)
        self.assertEqual((self.output / "results.xlsx").read_bytes(), workbook_bytes)
        self.assertEqual(json.loads((self.output / "report.json").read_text(encoding="utf-8"))["run"]["status"],
                         "cancelled")
        self.assertEqual({path.name for path in self.output.iterdir()}, original_names)

    def test_failed_json_replace_preserves_previous_report_and_workbook(self) -> None:
        write_report(self.output, self.plan, self.run)
        json_bytes = (self.output / "report.json").read_bytes()
        workbook_bytes = (self.output / "results.xlsx").read_bytes()
        original_names = {path.name for path in self.output.iterdir()}
        self.run["status"] = "cancelled"
        replace = Path.replace

        def fail_json_replace(path, target):
            if Path(target).name == "report.json":
                raise OSError("replacement denied")
            return replace(path, target)

        with patch.object(Path, "replace", new=fail_json_replace):
            with self.assertRaisesRegex(OSError, "replacement denied"):
                write_report(self.output, self.plan, self.run)
        self.assertEqual((self.output / "report.json").read_bytes(), json_bytes)
        self.assertEqual((self.output / "results.xlsx").read_bytes(), workbook_bytes)
        self.assertEqual({path.name for path in self.output.iterdir()}, original_names)


if __name__ == "__main__":
    unittest.main()
