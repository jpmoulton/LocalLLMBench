"""CPU-only report exports from persisted benchmark evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import load_workbook

from orchestrator.report import write_results, write_status


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
        report = write_results(self.output, self.plan, self.run)
        leaders = report["benchmark_leaders"]
        self.assertEqual(leaders["alpha"]["accuracy"]["trial_ids"], ["0001", "0002"])
        self.assertEqual(leaders["alpha"]["accuracy"]["value"], 0.75)
        self.assertEqual(leaders["alpha"]["loss"]["trial_ids"], ["0002"])
        self.assertEqual(leaders["alpha"]["loss"]["value"], 1.0)
        self.assertEqual(set(leaders["alpha"]), {"accuracy", "loss"})
        self.assertEqual(leaders["beta"], {})
        saved = json.loads((self.output / "results.json").read_text(encoding="utf-8"))
        workbook = self.workbook()
        self.assertEqual(workbook.sheetnames, ["Results"])
        results = self.rows(workbook["Results"])
        self.assertEqual(results, saved["results"])
        self.assertEqual(results[0]["score.accuracy"], 0.75)
        self.assertEqual(results[0]["performance.decode_tokens_per_second"], 100)
        self.assertEqual(results[2]["performance.decode_tokens_per_second"], 200)
        self.assertIsNone(results[1]["score.loss"])

    def test_all_combinations_and_missing_results_remain_visible(self) -> None:
        report = write_results(self.output, self.plan, self.run)
        rows = report["results"]
        self.assertEqual({(row["trial_id"], row["benchmark"]) for row in rows},
                         {(f"{i:04d}", name) for i in range(1, 6) for name in ("alpha", "beta")})
        self.assertEqual(rows[0]["benchmark_status"], "incomplete")
        self.assertEqual(rows[-1]["benchmark_status"], "pending")
        self.assertEqual(rows[0]["load_seconds"], 1.0)
        self.assertEqual(rows[1]["load_seconds"], 1.0)  # Same trial, not another load.
        self.assertEqual(rows[0]["trial_benchmark_seconds"], 8.0)
        self.assertIsNone(rows[-1]["trial_seconds"])
        self.assertEqual(rows[2]["trial_error"], "=failure")
        for row in rows:
            self.assertIsNone(row["performance.generated_tokens"])
            self.assertIsNone(row["performance.model_calls"])
        self.assertEqual(self.rows(self.workbook()["Results"]), rows)

    def test_aggregate_metrics_and_settings_match_without_sample_exports(self) -> None:
        summary = {
            "status": "incomplete", "denominator": 3, "scored": 1,
            "started_at": "2026-11-03T12:00:00+00:00",
            "finished_at": "2026-11-03T12:00:02.500000+00:00",
            "counts": {"pass": 1, "truncation": 1, "missing": 1},
            "performance": {
                "model_calls": 3, "latency_calls": 2, "decode_calls": 1, "prompt_calls": 0,
                "generated_tokens": 8, "decode_seconds": 0.5, "decode_tokens_per_second": 16,
                "decode_ms_per_token": 62.5, "request_latency_seconds_mean": 1.5,
            },
            "samples": [{"id": "private-task", "answer": "PRIVATE_ANSWER"}],
            "generation_samples": [{"id": "private-task", "reasoning": "PRIVATE_REASONING"}],
        }
        self.plan["trials"][0]["parameters"]["--override"] = {"args": ["layer.0", "cpu"]}
        self.plan["benchmark_options"] = {"alpha": {"selection": {"count": 3, "seed": 42}}}
        self.save_summary("0001", "alpha", summary)
        write_status(self.output, self.run)
        report = write_results(self.output, self.plan, self.run)
        result = report["results"][0]
        self.assertEqual(result["benchmark_seconds"], 2.5)
        self.assertEqual(result["performance.decode_tokens_per_second"], 16)
        self.assertEqual(result["performance.prompt_calls"], 0)
        self.assertIsNone(result["performance.prompt_tokens"])
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["scored_count"], 1)
        self.assertEqual(result["count.truncation"], 1)
        self.assertEqual(result["generation.max_tokens"], 32)
        self.assertEqual(result["inference.--parallel"], 2)
        self.assertEqual(result["inference.--override.args.0"], "layer.0")
        self.assertEqual(result["inference.--override.args.1"], "cpu")
        self.assertEqual(result["benchmark_options.selection.seed"], 42)
        self.assertEqual(result["request_concurrency"], 2)
        self.assertEqual(self.rows(self.workbook()["Results"]), report["results"])
        for row in report["results"]:
            self.assertTrue(all(not isinstance(value, (dict, list)) for value in row.values()))
        for name in ("results.json", "execution_report.json"):
            text = (self.output / name).read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_", text)
            self.assertNotIn("private-task", text)
        status = json.loads((self.output / "execution_report.json").read_text())
        self.assertNotIn("results", status)
        self.assertNotIn("durations", status["trials"][0])
        self.assertEqual(status["run"]["current_trial"], "0003")

    def test_external_strings_are_literal_and_match_json_cells(self) -> None:
        self.plan["trials"][0]["model"] = "=HYPERLINK(\"bad\")"
        self.plan["trials"][0]["parameters"]["--label"] = "text\x00end"
        self.save_summary("0001", "alpha", {
            "status": "completed", "score": 0.5,
            "score_metrics": {"score": {"direction": "higher"}},
        })
        report = write_results(self.output, self.plan, self.run)
        workbook = self.workbook()
        for sheet in workbook:
            for row in sheet:
                for cell in row:
                    self.assertNotEqual(cell.data_type, "f", (sheet.title, cell.coordinate))
        rows = self.rows(workbook["Results"])
        self.assertEqual(rows, report["results"])
        self.assertEqual(rows[0]["model"], "=HYPERLINK(\"bad\")")
        self.assertEqual(rows[0]["inference.--label"], "text\\x00end")
        self.assertEqual(rows[0]["score.score"], 0.5)
        self.assertEqual(rows[0]["trial_id"], "0001")

    def test_missing_boolean_nonfinite_and_undeclared_scores_never_win(self) -> None:
        metadata = {"score": {"direction": "higher"}}
        for trial_id, score in (("0001", float("nan")), ("0002", float("inf")),
                                ("0003", True), ("0004", "0.9"), ("0005", None)):
            self.save_summary(trial_id, "alpha", {
                "status": "completed", "score": score, "score_metrics": metadata,
            })
        report = write_results(self.output, self.plan, self.run)
        self.assertEqual(report["benchmark_leaders"]["alpha"], {})
        self.assertIsNone(report["results"][0]["score.score"])
        parsed = json.loads((self.output / "results.json").read_text(encoding="utf-8"),
                            parse_constant=lambda value: self.fail(f"Nonfinite JSON: {value}"))
        self.assertEqual(parsed["results"], report["results"])

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
                report = write_results(self.output, self.plan, self.run)
                self.assertEqual(report["benchmark_leaders"]["alpha"], {})

    def test_malformed_summary_is_visible_without_hiding_other_evaluations(self) -> None:
        broken = self.save_summary("0001", "alpha", {})
        broken.write_text("{unfinished", encoding="utf-8")
        self.save_summary("0002", "alpha", {
            "status": "completed", "score": 0,
            "score_metrics": {"score": {"direction": "higher"}},
        })
        report = write_results(self.output, self.plan, self.run)
        self.assertEqual(report["results"][0]["benchmark_status"], "incomplete")
        self.assertIn("JSONDecodeError", report["results"][0]["benchmark_error"])
        self.assertEqual(report["benchmark_leaders"]["alpha"]["score"]["value"], 0)
        self.assertEqual(report["benchmark_leaders"]["alpha"]["score"]["trial_ids"], ["0002"])

    def test_failed_workbook_save_preserves_previous_file_and_new_json(self) -> None:
        write_results(self.output, self.plan, self.run)
        workbook_bytes = (self.output / "results.xlsx").read_bytes()
        original_names = {path.name for path in self.output.iterdir()}
        self.run["trials"][0]["status"] = "cancelled"

        def fail_save(workbook, filename):
            self.assertEqual(Path(filename).suffix, ".xlsx")
            Path(filename).write_bytes(b"incomplete ZIP")
            raise OSError("disk full")

        with patch("orchestrator.report.Workbook.save", new=fail_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                write_results(self.output, self.plan, self.run)
        self.assertEqual((self.output / "results.xlsx").read_bytes(), workbook_bytes)
        self.assertEqual(json.loads((self.output / "results.json").read_text(encoding="utf-8"))["results"][0]["trial_status"],
                         "cancelled")
        self.assertEqual({path.name for path in self.output.iterdir()}, original_names)

    def test_failed_json_replace_preserves_previous_report_and_workbook(self) -> None:
        write_results(self.output, self.plan, self.run)
        json_bytes = (self.output / "results.json").read_bytes()
        workbook_bytes = (self.output / "results.xlsx").read_bytes()
        original_names = {path.name for path in self.output.iterdir()}
        self.run["trials"][0]["status"] = "cancelled"
        replace = Path.replace

        def fail_json_replace(path, target):
            if Path(target).name == "results.json":
                raise OSError("replacement denied")
            return replace(path, target)

        with patch.object(Path, "replace", new=fail_json_replace):
            with self.assertRaisesRegex(OSError, "replacement denied"):
                write_results(self.output, self.plan, self.run)
        self.assertEqual((self.output / "results.json").read_bytes(), json_bytes)
        self.assertEqual((self.output / "results.xlsx").read_bytes(), workbook_bytes)
        self.assertEqual({path.name for path in self.output.iterdir()}, original_names)


if __name__ == "__main__":
    unittest.main()
