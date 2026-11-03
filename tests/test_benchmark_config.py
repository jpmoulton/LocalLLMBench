"""Regression tests for explicit benchmark selection and shared configuration."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from benchmarks.config import DEFAULT_CONFIG, load_config, resolve_config
from benchmarks import registry


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        replacement = patch.dict(
            registry.BENCHMARKS,
            {"alpha": "test_benchmarks.alpha", "beta": "test_benchmarks.beta"},
            clear=True,
        )
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_default_selection_is_all_registered_names_in_order(self) -> None:
        self.assertEqual(registry.resolve_benchmarks(None), ["alpha", "beta"])
        registry.BENCHMARKS["gamma"] = "test_benchmarks.gamma"
        self.assertEqual(registry.resolve_benchmarks(None), ["alpha", "beta", "gamma"])

    def test_explicit_selection_preserves_order_and_excludes_other_names(self) -> None:
        self.assertEqual(registry.resolve_benchmarks(["beta"]), ["beta"])
        self.assertEqual(registry.resolve_benchmarks(["beta", "alpha"]), ["beta", "alpha"])

    def test_selection_is_returned_independently(self) -> None:
        names = ["alpha"]
        resolved = registry.resolve_benchmarks(names)
        resolved.append("beta")
        self.assertEqual(names, ["alpha"])
        all_names = registry.resolve_benchmarks(None)
        all_names.clear()
        self.assertEqual(list(registry.BENCHMARKS), ["alpha", "beta"])

    def test_empty_unknown_duplicate_and_nonstring_names_fail(self) -> None:
        for names in (
            [], ["missing"], ["alpha", "alpha"], [1], [None], [True], [{}], [[]],
            [""], [" "], "alpha", ("alpha",), {}, False,
        ):
            with self.subTest(names=names), self.assertRaises(ValueError):
                registry.resolve_benchmarks(names)



class ConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def load_overrides(self, overrides: object) -> dict:
        path = self.root / "config.json"
        path.write_text(json.dumps(overrides), encoding="utf-8")
        return load_config(path)


class SharedConfigTests(ConfigTestCase):
    def setUp(self) -> None:
        super().setUp()
        replacement = patch.dict(registry.BENCHMARKS, {}, clear=True)
        replacement.start()
        self.addCleanup(replacement.stop)
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        self.alpha = self.add_definition(
            "alpha", {"input_path": "missing-alpha", "sampling": {"count": 4, "seed": 42}}
        )
        self.beta = self.add_definition("beta", {"input_path": "missing-beta", "mode": "standard"})

    def add_definition(self, name: str, defaults: dict) -> ModuleType:
        module_path = f"test_benchmarks.{name}"
        definition = ModuleType(f"{module_path}.definition")
        definition.DEFAULT_OPTIONS = defaults
        definition.PATH_OPTIONS = ("input_path",)
        definition.NEEDS_DOCKER = False
        definition.validate_options = Mock()
        definition.prepare = Mock(side_effect=AssertionError("Config must not load datasets"))
        registry.BENCHMARKS[name] = module_path
        sys.modules[definition.__name__] = definition
        return definition

    def test_omitted_and_null_selection_load_all_current_definitions(self) -> None:
        self.assertEqual(self.load_overrides({})["benchmarks"], ["alpha", "beta"])
        gamma = self.add_definition("gamma", {"new_option": {"value": 3}})
        for overrides in ({}, {"benchmarks": None}):
            with self.subTest(overrides=overrides):
                config = self.load_overrides(overrides)
                self.assertEqual(config["benchmarks"], ["alpha", "beta", "gamma"])
                self.assertEqual(config["benchmark_options"]["gamma"], gamma.DEFAULT_OPTIONS)

    def test_explicit_subset_and_order_are_preserved(self) -> None:
        for names in (["beta"], ["beta", "alpha"]):
            with self.subTest(names=names):
                self.assertEqual(self.load_overrides({"benchmarks": names})["benchmarks"], names)

    def test_invalid_benchmark_selections_fail_during_config_load(self) -> None:
        for names in ([], ["missing"], ["alpha", "alpha"], [1], [None], [""], "alpha"):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.load_overrides({"benchmarks": names})

    def test_recursive_overrides_and_returned_configs_are_isolated(self) -> None:
        original_defaults = copy.deepcopy(DEFAULT_CONFIG)
        original_options = copy.deepcopy(self.alpha.DEFAULT_OPTIONS)
        baseline = self.load_overrides({})
        overrides = {
            "benchmark_options": {"alpha": {"sampling": {"seed": 773}}},
            "generation": {"thinking": False},
        }
        original_overrides = copy.deepcopy(overrides)
        changed = self.load_overrides(overrides)
        expected = copy.deepcopy(baseline)
        expected["benchmark_options"]["alpha"]["sampling"]["seed"] = 773
        expected["generation"]["thinking"] = False
        self.assertEqual(changed, expected)
        self.assertEqual(overrides, original_overrides)
        changed["benchmark_options"]["alpha"]["sampling"]["count"] = 1
        changed["generation"]["top_p"] = 0.25
        changed["benchmarks"].clear()
        self.assertEqual(self.load_overrides({}), baseline)
        self.assertEqual(DEFAULT_CONFIG, original_defaults)
        self.assertEqual(self.alpha.DEFAULT_OPTIONS, original_options)

    def test_in_memory_resolution_does_not_retain_override_references(self) -> None:
        self.alpha.DEFAULT_OPTIONS["labels"] = []
        overrides = {
            "benchmarks": ["beta", "alpha"],
            "benchmark_options": {"alpha": {"labels": ["first", "second"]}},
        }
        original = copy.deepcopy(overrides)
        resolved = resolve_config(overrides)
        resolved["benchmarks"].clear()
        resolved["benchmark_options"]["alpha"]["labels"].clear()
        self.assertEqual(overrides, original)
        self.assertEqual(resolve_config(overrides), self.load_overrides(overrides))

    def test_unknown_keys_fail_at_root_and_nested_levels(self) -> None:
        for overrides in (
            {"benchmrks": []},
            {"generation": {"temperatur": 1}},
            {"benchmark_options": {"missing": {}}},
            {"benchmark_options": {"alpha": {"other": 42}}},
            {"benchmark_options": {"alpha": {"sampling": {"random_seed": 42}}}},
            {"execution": {"retry": 1}},
            {"inference": {"context": 100}},
            {"dataset_path": "old-schema"},
            {"selection": {"count": 1}},
            {"execution": {"test_timeout": 60}},
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "Unknown"):
                self.load_overrides(overrides)

    def test_config_types_and_numeric_ranges_are_enforced(self) -> None:
        invalid = [
            [], None,
            {"benchmark_options": None},
            {"benchmark_options": {"alpha": None}},
            {"benchmark_options": {"alpha": {"sampling": None}}},
            {"inference": None},
            {"generation": None},
            {"execution": None},
            {"output_dir": 123},
            {"output_dir": " "},
            {"inference": {"model": ""}},
            {"inference": {"container": None}},
            {"generation": {"thinking": 1}},
            {"generation": {"max_tokens": 0}},
            {"generation": {"seed": False}},
            {"generation": {"top_k": -1}},
            {"generation": {"temperature": -0.1}},
            {"generation": {"temperature": 2.1}},
            {"generation": {"temperature": float("nan")}},
            {"generation": {"temperature": 10 ** 1000}},
            {"generation": {"top_p": 0}},
            {"generation": {"top_p": 1.1}},
            {"generation": {"min_p": -0.1}},
            {"generation": {"min_p": 1.1}},
            {"execution": {"concurrency": True}},
            {"execution": {"concurrency": 0}},
            {"execution": {"request_timeout": 0}},
            {"execution": {"request_timeout": float("inf")}},
            {"execution": {"request_timeout": False}},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.load_overrides(overrides)

    def test_generation_and_execution_valid_boundaries_are_retained(self) -> None:
        overrides = {
            "generation": {
                "thinking": False, "max_tokens": 1, "seed": -1,
                "top_k": 0, "temperature": 0, "top_p": 1, "min_p": 0,
            },
            "execution": {"concurrency": 1, "request_timeout": 0.5},
        }
        config = self.load_overrides(overrides)
        self.assertEqual(config["generation"], overrides["generation"])
        self.assertEqual(config["execution"], overrides["execution"])

    def test_unselected_inputs_are_never_read_and_runtime_is_not_imported(self) -> None:
        path = self.root / "config.json"
        path.write_text(json.dumps({"benchmarks": ["beta"]}), encoding="utf-8")
        read_text = Path.read_text

        def read_config_only(target: Path, *args, **kwargs):
            self.assertEqual(target, path, "Config loading must not read a benchmark input")
            return read_text(target, *args, **kwargs)

        with (
            patch.object(Path, "read_text", read_config_only),
            patch.object(Path, "read_bytes", side_effect=AssertionError("No dataset reads")),
            patch.object(Path, "exists", side_effect=AssertionError("No dataset existence checks")),
        ):
            config = load_config(path)
        self.assertEqual(config["benchmarks"], ["beta"])
        self.assertEqual(config["benchmark_options"]["alpha"]["input_path"], "missing-alpha")


class BigCodeBenchOptionsTests(ConfigTestCase):

    def test_invalid_benchmark_owned_options_are_rejected(self) -> None:
        for options in (
            {"dataset_path": " "},
            {"dataset_path": None},
            {"selection": None},
            {"selection": {"count": 0}},
            {"selection": {"count": True}},
            {"selection": {"seed": 1.5}},
            {"test_timeout": float("inf")},
            {"test_timeout": False},
            {"test_timeout": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load_overrides({"benchmark_options": {"bigcodebench": options}})


if __name__ == "__main__":
    unittest.main()
