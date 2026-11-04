"""Stdlib regressions for frozen retrieval inputs, recall, and coverage boundaries."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.ruler.definition import (
    DEFAULT_OPTIONS, GENERATOR_REPOSITORY, GENERATOR_REVISION, PROTOCOL,
    SCORE_METRICS, TASKS, prepare, validate_options,
)
from benchmarks.ruler.scoring import (
    SCORER_NAME, normalize_prediction, score_output, substring_recall, summarize,
)


def corpus_records(count: int = 8, tasks: tuple = TASKS) -> list[dict]:
    return [
        {
            "id": f"{task}/{index}", "task": task, "index": index,
            "input": f"Remember café {task} {index}.\n", "answer_prefix": "Answer: ",
            "outputs": [f"answer-{index}", "another answer"], "input_tokens": 32,
        }
        for task in tasks for index in range(count)
    ]


def model_output(answer: str = "", reason: str = "stop", error=None):
    return SimpleNamespace(
        completion=answer, error=error, usage=None,
        choices=[SimpleNamespace(
            stop_reason=reason,
            message=SimpleNamespace(content=[
                SimpleNamespace(type="reasoning", reasoning="hidden reference"),
                SimpleNamespace(type="text", text=answer),
            ]),
        )],
    )


def sample_result(record: dict, recall: float = 1.0, reason: str = "stop", **changes):
    result = SimpleNamespace(
        id=record["id"], metadata={"task": record["task"]},
        output=model_output(reason=reason),
        scores={SCORER_NAME: SimpleNamespace(value=recall)}, error=None, limit=None,
    )
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def eval_log(samples: list, status: str = "success", error=None):
    return SimpleNamespace(location="inspect/test.eval", status=status, error=error, samples=samples)


class RulerCorpusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_corpus(self, records=None, name="dataset", manifest_changes=None):
        records = corpus_records() if records is None else records
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        raw = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
        (directory / "corpus.jsonl").write_bytes(raw)
        manifest = {
            "schema_version": 1, "protocol": PROTOCOL,
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "generator": {
                "repository": GENERATOR_REPOSITORY, "revision": GENERATOR_REVISION,
                "files_sha256": {"scripts/data/synthetic/niah.py": "a" * 64},
            },
            "tokenizer": {"encoding": "cl100k_base", "package": "tiktoken", "version": "0.8.0"},
            "generation": {
                "seed": 42, "input_tokens": 65536, "reserved_output_tokens": 8192,
                "samples_per_task": max((record.get("index", 0) for record in records if type(record.get("index")) is int), default=0) + 1,
                "tasks": list(TASKS),
            },
            "essays": {"files": [{"name": "essay.txt", "sha256": "b" * 64}], "separator": "\n\n"},
            "dependencies": {"tiktoken": "0.8.0"},
        }
        manifest.update(manifest_changes or {})
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def options(self, directory, count=8, seed=42, tasks=TASKS):
        options = copy.deepcopy(DEFAULT_OPTIONS)
        options.update(dataset_path=str(directory), tasks=list(tasks))
        options["selection"] = {"count_per_task": count, "seed": seed}
        return options

    def test_default_has_64_unchanged_prompts_from_all_eight_variants(self):
        records = corpus_records()
        selected, provenance = prepare(self.options(self.write_corpus(records)))
        self.assertEqual(len(selected), 64)
        self.assertEqual(provenance["tasks"], list(TASKS))
        originals = {record["id"]: record for record in records}
        for record in selected:
            self.assertEqual(record, originals[record["id"]])
        for task in TASKS:
            self.assertEqual(sum(record["task"] == task for record in selected), 8)
        self.assertEqual(provenance["selected_ids"], [record["id"] for record in selected])

    def test_selection_and_full_provenance_are_path_independent_and_repeatable(self):
        first = self.write_corpus(name="host")
        second = self.write_corpus(name="container")
        prepared = prepare(self.options(first))
        self.assertEqual(prepared, prepare(self.options(second)))
        self.assertEqual(prepared, prepare(self.options(first)))
        self.assertNotIn(str(first), json.dumps(prepared[1]))

    def test_row_order_does_not_change_selected_records(self):
        records = corpus_records(16)
        first, first_provenance = prepare(self.options(self.write_corpus(records, "first")))
        second, second_provenance = prepare(self.options(self.write_corpus(list(reversed(records)), "second")))
        self.assertEqual(first, second)
        self.assertNotEqual(first_provenance["corpus_sha256"], second_provenance["corpus_sha256"])

    def test_count_prefixes_and_configured_task_order(self):
        directory = self.write_corpus(corpus_records(16))
        selected_tasks = (TASKS[-1], TASKS[0])
        full, _ = prepare(self.options(directory, count=16, tasks=selected_tasks))
        smaller, provenance = prepare(self.options(directory, count=3, tasks=selected_tasks))
        self.assertEqual(smaller, full[:3] + full[16:19])
        self.assertEqual(provenance["tasks"], list(selected_tasks))

    def test_selection_seed_is_distinct_from_generation_seed(self):
        directory = self.write_corpus(corpus_records(32))
        first, first_provenance = prepare(self.options(directory, count=4))
        second, second_provenance = prepare(self.options(directory, count=4, seed=43))
        self.assertNotEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertEqual(first_provenance["generation_seed"], second_provenance["generation_seed"])
        self.assertEqual(second_provenance["selection_seed"], 43)

    def test_invalid_options_are_rejected(self):
        baseline = self.options(self.root / "unused")
        mutations = [
            ("tasks", []), ("tasks", [TASKS[0], TASKS[0]]), ("tasks", ["qa"]),
            ("tasks", [None]), ("dataset_path", " "), ("unknown", 1),
            ("selection", {"count_per_task": 8, "seed": 42, "extra": True}),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_options({**baseline, key: value})
        for count in (0, -1, True, 1.5, "8", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_options(self.options(self.root, count=count))
        for seed in (True, 1.5, "42", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_options(self.options(self.root, seed=seed))

    def test_hash_mismatch_and_missing_manifest_are_rejected(self):
        directory = self.write_corpus()
        corpus = directory / "corpus.jsonl"
        corpus.write_bytes(corpus.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "corpus_sha256"):
            prepare(self.options(directory))
        (directory / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(self.options(directory))

    def test_bad_manifest_identity_and_budget_are_rejected(self):
        for changes in (
            {"schema_version": True}, {"protocol": "other"}, {"corpus_sha256": None},
            {"generator": {}}, {"tokenizer": {}}, {"generation": {}},
            {"essays": {}}, {"dependencies": {}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare(self.options(self.write_corpus(manifest_changes=changes)))

    def test_each_selected_variant_requires_enough_records(self):
        records = corpus_records()
        directory = self.write_corpus(records[:-1])
        with self.assertRaisesRegex(ValueError, "niah_multiquery.*need 8.*found 7"):
            prepare(self.options(directory))
        selected, _ = prepare(self.options(directory, tasks=(TASKS[0],)))
        self.assertEqual(len(selected), 8)
        with self.assertRaises(ValueError):
            prepare(self.options(self.write_corpus([])))

    def test_duplicate_ids_and_task_indices_are_rejected_even_outside_subset(self):
        records = corpus_records()
        records.append(copy.deepcopy(records[-1]))
        with self.assertRaisesRegex(ValueError, "Duplicate sample id"):
            prepare(self.options(self.write_corpus(records), count=1, tasks=(TASKS[0],)))
        records[-1]["id"] = "a-different-id"
        with self.assertRaisesRegex(ValueError, "Duplicate task/index"):
            prepare(self.options(self.write_corpus(records), count=1))

    def test_invalid_records_are_rejected_even_outside_selected_task(self):
        mutations = [
            ("id", ""), ("task", "qa"), ("index", True), ("index", -1),
            ("input", ""), ("answer_prefix", None), ("outputs", []),
            ("outputs", [""]), ("outputs", ["valid", None]), ("outputs", "answer"),
            ("input_tokens", 65537), ("input_tokens", 0), ("input_tokens", True),
        ]
        for key, value in mutations:
            records = corpus_records()
            records[-1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prepare(self.options(self.write_corpus(records), count=1, tasks=(TASKS[0],)))

    def test_prompt_change_changes_frozen_identity(self):
        records = corpus_records()
        _, original = prepare(self.options(self.write_corpus(records, "original")))
        records[0]["input"] += "Changed content."
        _, changed = prepare(self.options(self.write_corpus(records, "changed")))
        self.assertEqual(original["selected_ids"], changed["selected_ids"])
        self.assertNotEqual(original, changed)


class RulerScoringTests(unittest.TestCase):
    def test_fractional_substrings_case_and_duplicate_references(self):
        self.assertEqual(substring_recall("prefixALPHAsuffix", ["alpha", "beta"]), 0.5)
        self.assertAlmostEqual(substring_recall("ALPHA", ["alpha", "alpha", "missing"]), 2 / 3)
        self.assertEqual(substring_recall("wrong", ["right"]), 0)
        self.assertEqual(substring_recall("", ["right"]), 0)
        self.assertEqual(substring_recall("first and second", ["SECOND", "FIRST"]), 1)

    def test_official_control_character_normalization(self):
        self.assertEqual(normalize_prediction(" \x00ALPHA\tBETA\x1f \n"), "alpha\nbeta")
        self.assertEqual(substring_recall("A\tB", ["a b", "a\nb"]), 0.5)
        self.assertEqual(substring_recall("x\x7fy", ["x\x7fy"]), 1)
        # lower(), rather than Unicode casefold(), is the upstream protocol.
        self.assertEqual(substring_recall("STRASSE", ["straße"]), 0)

    def test_invalid_references_never_produce_a_vacuous_score(self):
        for references in ([], [""], [None], "answer"):
            with self.subTest(references=references), self.assertRaises(ValueError):
                substring_recall("answer", references)

    def test_final_output_excludes_separately_represented_reasoning(self):
        output = model_output("final answer")
        answer, recall = score_output(output, ["hidden reference", "FINAL"])
        self.assertEqual(answer, "final answer")
        self.assertEqual(recall, 0.5)
        output.completion = ""
        self.assertEqual(score_output(output, ["hidden reference"]), ("", 0.0))

    def test_failed_and_truncated_generation_get_no_score(self):
        for reason in ("max_tokens", "model_length", "unknown", "tool_calls"):
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                score_output(model_output("right", reason), ["right"])
        with self.assertRaises(RuntimeError):
            score_output(model_output("right", error="request failed"), ["right"])
        output = model_output("right")
        output.choices = []
        with self.assertRaises(RuntimeError):
            score_output(output, ["right"])


class RulerSummaryTests(unittest.TestCase):
    def setUp(self):
        self.records = corpus_records(2, (TASKS[0], TASKS[-1]))
        self.provenance = {
            "protocol": PROTOCOL, "selected_ids": [record["id"] for record in self.records],
            "sample_tasks": {record["id"]: record["task"] for record in self.records},
            "tasks": [TASKS[0], TASKS[-1]], "count_per_task": 2,
            "input_tokens": 65536, "reserved_output_tokens": 8192,
        }

    def samples(self, values=(1.0, 0.5, 0.0, 0.5)):
        return [sample_result(record, value) for record, value in zip(self.records, values)]

    def assert_incomplete(self, result):
        self.assertEqual(result["status"], "incomplete")
        for key in SCORE_METRICS:
            self.assertIsNone(result[key], key)
        for result_by_task in result["per_task"].values():
            self.assertIsNone(result_by_task["score"])

    def test_complete_fractional_and_incorrect_answers_are_completed(self):
        result = summarize([eval_log(self.samples())], self.provenance, None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["denominator"], 4)
        self.assertEqual(result["scored"], 4)
        self.assertEqual(result["retrieval_score"], 50)
        self.assertEqual(result[f"{TASKS[0]}_score"], 75)
        self.assertEqual(result[f"{TASKS[-1]}_score"], 25)
        self.assertIsNone(result[f"{TASKS[1]}_score"])
        self.assertEqual(result["counts"]["partial"], 2)
        self.assertEqual(result["counts"]["fail"], 1)
        self.assertEqual([sample["id"] for sample in result["samples"]], self.provenance["selected_ids"])
        json.dumps(result, allow_nan=False)

    def test_all_wrong_is_a_complete_zero_not_an_execution_failure(self):
        result = summarize([eval_log(self.samples((0, 0, 0, 0)))], self.provenance, None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["retrieval_score"], 0)
        self.assertEqual(result["scored"], 4)

    def test_missing_or_empty_logs_withhold_scores(self):
        for logs in ([], [eval_log([])], [eval_log(self.samples()[:-1])]):
            with self.subTest(logs=logs):
                result = summarize(logs, self.provenance, None)
                self.assert_incomplete(result)
                self.assertGreater(result["counts"]["missing"], 0)
                self.assertEqual(result["denominator"], 4)

    def test_duplicate_and_unexpected_samples_withhold_every_aggregate(self):
        duplicate = self.samples()[0]
        unexpected = sample_result({"id": "unselected/0", "task": TASKS[0]})
        for extra in (duplicate, unexpected):
            with self.subTest(extra=extra.id):
                result = summarize([eval_log(self.samples() + [extra])], self.provenance, None)
                self.assert_incomplete(result)
                self.assertEqual(result["scored"], 4)
                self.assertTrue(result["inspect_log_errors"])
        result = summarize([eval_log(self.samples()), eval_log([duplicate])], self.provenance, None)
        self.assert_incomplete(result)

    def test_sample_error_model_error_and_limit_withhold_scores(self):
        for field, value in (("error", {"message": "failure"}), ("limit", {"type": "time"})):
            samples = self.samples()
            setattr(samples[0], field, value)
            result = summarize([eval_log(samples)], self.provenance, None)
            self.assert_incomplete(result)
            self.assertEqual(result["counts"]["error"], 1)
            self.assertEqual(result["scored"], 3)
        samples = self.samples()
        samples[0].output.error = "connection lost"
        result = summarize([eval_log(samples)], self.provenance, None)
        self.assert_incomplete(result)
        self.assertEqual(result["samples"][0]["model_error"], "connection lost")

    def test_truncated_sample_does_not_become_an_incorrect_answer(self):
        for reason in ("max_tokens", "model_length"):
            samples = self.samples()
            samples[0].output.choices[0].stop_reason = reason
            result = summarize([eval_log(samples)], self.provenance, None)
            self.assert_incomplete(result)
            self.assertEqual(result["counts"]["truncation"], 1)
            self.assertEqual(result["counts"]["fail"], 1)
            self.assertEqual(result["scored"], 3)
            self.assertIsNone(result["samples"][0]["recall"])

    def test_missing_ambiguous_or_invalid_scores_withhold_aggregates(self):
        for scores in (
            None, {}, {"different_scorer": SimpleNamespace(value=1)},
            {SCORER_NAME: SimpleNamespace(value=1), "extra": SimpleNamespace(value=1)},
            *({SCORER_NAME: SimpleNamespace(value=value)} for value in (True, -1, 1.1, "1", float("nan"))),
        ):
            with self.subTest(scores=scores):
                samples = self.samples()
                samples[0].scores = scores
                result = summarize([eval_log(samples)], self.provenance, None)
                self.assert_incomplete(result)
                self.assertEqual(result["counts"]["error"], 1)
                json.dumps(result, allow_nan=False)

    def test_log_failure_run_error_and_metadata_mismatch_preserve_evidence(self):
        failed = eval_log(self.samples(), status="error", error={"message": "log failure"})
        result = summarize([failed], self.provenance, None)
        self.assert_incomplete(result)
        self.assertEqual(result["inspect_log_errors"][0]["error"], failed.error)
        run_error = {"type": "RuntimeError", "message": "log recovery failed"}
        result = summarize([eval_log(self.samples())], self.provenance, run_error)
        self.assert_incomplete(result)
        self.assertEqual(result["run_error"], run_error)
        samples = self.samples()
        samples[0].metadata["task"] = TASKS[1]
        self.assert_incomplete(summarize([eval_log(samples)], self.provenance, None))

    def test_multiple_successful_logs_and_out_of_order_results_preserve_coverage(self):
        samples = list(reversed(self.samples()))
        result = summarize([eval_log(samples[:2]), eval_log(samples[2:])], self.provenance, None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["retrieval_score"], 50)
        self.assertEqual(len(result["inspect_logs"]), 2)


if __name__ == "__main__":
    unittest.main()
