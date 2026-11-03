"""Regression tests for corpus identity, subset policy, and config isolation."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.config import load_config
from benchmarks.bigcodebench.definition import prepare, select_dataset


def corpus_records(count: int = 128) -> list[dict]:
    return [
        {
            "task_id": f"BigCodeBench/{index}",
            "complete_prompt": f"def task_func():\n    \"\"\"Return {index}; café.\"\"\"\n",
            "canonical_solution": f"    return {index}\n",
            "test": f"assert task_func() == {index}\n",
            "libs": "[]",
        }
        for index in range(count)
    ]


class BigCodeBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_corpus(self, records: list[dict], name: str = "dataset") -> Path:
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        raw = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode(
            "utf-8"
        )
        (directory / "corpus.jsonl").write_bytes(raw)
        manifest = {
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset_version": "v0.1.4",
            "source": {"dataset_revision": "test-revision", "dataset_split": "v0.1.4"},
            # These describe an unrelated old selection and must not control ours.
            "count": 1,
            "seed": 999,
            "selected_ids": ["old-policy-task"],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "sample.jsonl").write_text("not JSON and not the corpus", encoding="utf-8")
        return directory

    def load_overrides(self, overrides: object) -> dict:
        path = self.root / "config.json"
        path.write_text(json.dumps(overrides), encoding="utf-8")
        return load_config(path)


    def test_row_and_object_key_order_do_not_change_selection_or_fingerprint(self) -> None:
        records = corpus_records()
        reordered = [dict(reversed(list(record.items()))) for record in reversed(records)]
        first = self.write_corpus(records, "first")
        second = self.write_corpus(reordered, "second")
        selected_a, provenance_a = select_dataset(first, 64, 42)
        selected_b, provenance_b = select_dataset(second, 64, 42)
        self.assertEqual(selected_a, selected_b)
        self.assertNotEqual(provenance_a["corpus_sha256"], provenance_b["corpus_sha256"])
        self.assertEqual(
            provenance_a["corpus_fingerprint_sha256"], provenance_b["corpus_fingerprint_sha256"]
        )

    def test_fingerprint_includes_record_content_not_only_ids(self) -> None:
        records = corpus_records(8)
        first = self.write_corpus(records, "first")
        records[0]["canonical_solution"] = "    return 999\n"
        second = self.write_corpus(records, "second")
        _, original = select_dataset(first, 4, 42)
        _, changed = select_dataset(second, 4, 42)
        self.assertEqual(original["selected_ids"], changed["selected_ids"])
        self.assertNotEqual(original["corpus_fingerprint_sha256"], changed["corpus_fingerprint_sha256"])

    def test_alternate_seeds_change_subset_membership(self) -> None:
        directory = self.write_corpus(corpus_records())
        _, first = select_dataset(directory, 16, 42)
        _, second = select_dataset(directory, 16, 43)
        self.assertNotEqual(set(first["selected_ids"]), set(second["selected_ids"]))
        self.assertEqual(select_dataset(directory, 16, 42)[1], first)

    def test_counts_select_nested_prefixes_including_full_population(self) -> None:
        directory = self.write_corpus(corpus_records())
        full, _ = select_dataset(directory, 128, 42)
        for count in (1, 16, 64, 127):
            with self.subTest(count=count):
                selected, _ = select_dataset(directory, count, 42)
                self.assertEqual(selected, full[:count])

    def test_generation_and_model_overrides_do_not_change_selection(self) -> None:
        directory = self.write_corpus(corpus_records())
        baseline = self.load_overrides({
            "benchmark_options": {"bigcodebench": {"dataset_path": str(directory)}}
        })
        changed = self.load_overrides(
            {
                "benchmark_options": {"bigcodebench": {"dataset_path": str(directory)}},
                "generation": {
                    "seed": 123456,
                    "thinking": False,
                    "max_tokens": 100,
                    "temperature": 0,
                    "top_k": 0,
                    "top_p": 0.5,
                    "min_p": 0,
                },
                "inference": {"model": "a-different-model", "minimum_context": 262144},
            }
        )
        self.assertNotEqual(baseline["generation"], changed["generation"])
        self.assertEqual(
            prepare(baseline["benchmark_options"]["bigcodebench"]),
            prepare(changed["benchmark_options"]["bigcodebench"]),
        )

    def test_invalid_selection_counts_and_seeds_are_rejected(self) -> None:
        directory = self.write_corpus(corpus_records(8))
        for count in (0, -1, 9, True, 1.5, "2", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_dataset(directory, count, 42)
        for seed in (True, 1.5, "42", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                select_dataset(directory, 1, seed)
        empty = self.write_corpus([], "empty")
        with self.assertRaises(ValueError):
            select_dataset(empty, 1, 42)

    def test_duplicate_ids_are_rejected_even_outside_selected_prefix(self) -> None:
        records = corpus_records(8)
        records.append(copy.deepcopy(records[-1]))
        directory = self.write_corpus(records)
        with self.assertRaisesRegex(ValueError, "Duplicate task_id"):
            select_dataset(directory, 1, 42)

    def test_missing_and_invalid_required_fields_are_rejected(self) -> None:
        for field in ("task_id", "complete_prompt", "canonical_solution", "test", "libs"):
            for value in (None, "", 123):
                with self.subTest(field=field, value=value):
                    records = corpus_records(8)
                    if value is None:
                        del records[-1][field]
                    else:
                        records[-1][field] = value
                    directory = self.write_corpus(records)
                    with self.assertRaisesRegex(ValueError, field):
                        select_dataset(directory, 1, 42)

    def test_stale_manifest_hash_is_rejected(self) -> None:
        directory = self.write_corpus(corpus_records(8))
        corpus = directory / "corpus.jsonl"
        corpus.write_bytes(corpus.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "corpus_sha256"):
            select_dataset(directory, 1, 42)

    def test_manifest_without_hash_or_without_file_is_supported(self) -> None:
        directory = self.write_corpus(corpus_records(8))
        path = directory / "manifest.json"
        path.write_text('{"source":{"dataset_split":"custom","dataset_revision":"revision"}}')
        selected, provenance = select_dataset(directory, 2, 42)
        self.assertEqual(provenance["dataset_version"], "custom")
        self.assertEqual(provenance["dataset_revision"], "revision")
        path.unlink()
        without_manifest, provenance = select_dataset(directory, 2, 42)
        self.assertEqual(without_manifest, selected)
        self.assertIsNone(provenance["dataset_version"])
        self.assertIsNone(provenance["dataset_revision"])


if __name__ == "__main__":
    unittest.main()
