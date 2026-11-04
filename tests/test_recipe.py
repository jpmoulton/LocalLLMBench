"""Host-only recipe parsing, trial expansion, and fixed workload regressions."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import yaml

from benchmarks import registry
from benchmarks.bigcodebench import definition
from benchmarks.config import resolve_config
from orchestrator.recipe import build_plan


class RecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        # These recipe tests supply only a BigCodeBench corpus; plugin inputs stay isolated.
        self.enterContext(patch.dict(
            registry.BENCHMARKS, {"bigcodebench": "benchmarks.bigcodebench"}, clear=True,
        ))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.recipe_dir = self.root / "recipes"
        self.recipe_dir.mkdir()
        self.dataset = self.recipe_dir / "dataset"
        self.dataset.mkdir()
        records = [
            {
                "task_id": f"BigCodeBench/{index}",
                "complete_prompt": "def task_func():\n",
                "canonical_solution": f"    return {index}\n",
                "test": f"assert task_func() == {index}",
                "libs": [],
            }
            for index in range(8)
        ]
        (self.dataset / "corpus.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        for name in ("first model.gguf", "second.gguf"):
            (self.recipe_dir / name).write_bytes(b"test model path; never loaded")
        self.recipe = {
            "schema_version": 1,
            "models": ["first model.gguf"],
            "inference": {"configurations": [{"parameters": {"--ctx-size": 4096}}]},
            "benchmark_options": {
                "bigcodebench": {"dataset_path": "dataset", "selection": {"count": 3}}
            },
        }

    def plan(self, recipe: dict | None = None) -> dict:
        path = self.recipe_dir / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(self.recipe if recipe is None else recipe, sort_keys=False),
            encoding="utf-8",
        )
        return build_plan(recipe_path=path, cwd=self.root)

    def test_models_groups_and_sorted_axes_have_stable_expansion_order(self) -> None:
        self.recipe["models"].append("second.gguf")
        self.recipe["inference"]["configurations"] = [
            {
                "name": "paired",
                "parameters": {"--threads": [2, 4], "--ctx-size": [4096, 8192]},
                "kv_quantization": ["q8_0", {"k": "q8_0", "v": "q4_0"}],
            },
            {"parameters": {"--ctx-size": 16384}},
        ]
        plan = self.plan()
        first_model = str(self.recipe_dir / "first model.gguf")
        second_model = str(self.recipe_dir / "second.gguf")
        self.assertEqual([trial["model"] for trial in plan["trials"]], [first_model] * 9 + [second_model] * 9)
        self.assertEqual([trial["id"] for trial in plan["trials"]], [f"{i:04d}" for i in range(1, 19)])
        self.assertEqual(
            [
                (trial["parameters"]["--ctx-size"], trial["parameters"]["--threads"],
                 trial["parameters"]["--cache-type-k"], trial["parameters"]["--cache-type-v"])
                for trial in plan["trials"][:8]
            ],
            [
                (4096, 2, "q8_0", "q8_0"), (4096, 2, "q8_0", "q4_0"),
                (4096, 4, "q8_0", "q8_0"), (4096, 4, "q8_0", "q4_0"),
                (8192, 2, "q8_0", "q8_0"), (8192, 2, "q8_0", "q4_0"),
                (8192, 4, "q8_0", "q8_0"), (8192, 4, "q8_0", "q4_0"),
            ],
        )
        self.assertEqual(plan["trials"][8]["group"], "group-2")
        self.assertEqual(plan["trials"][8]["parameters"], {"--ctx-size": 16384})
        self.recipe["inference"]["configurations"][0]["parameters"] = {
            "--ctx-size": [4096, 8192], "--threads": [2, 4]
        }
        self.assertEqual(self.plan()["trials"], plan["trials"])

    def test_native_cache_axes_are_independent_and_equivalent_groups_are_kept(self) -> None:
        group = {"parameters": {
            "--ctx-size": 4096,
            "--cache-type-v": ["q4_0", "q8_0"],
            "--cache-type-k": ["f16", "q8_0"],
        }}
        self.recipe["inference"]["configurations"] = [group, copy.deepcopy(group)]
        trials = self.plan()["trials"]
        self.assertEqual(len(trials), 8)
        self.assertEqual(
            [(t["parameters"]["--cache-type-k"], t["parameters"]["--cache-type-v"]) for t in trials[:4]],
            [("f16", "q4_0"), ("f16", "q8_0"), ("q8_0", "q4_0"), ("q8_0", "q8_0")],
        )
        self.assertEqual(trials[0]["parameters"], trials[4]["parameters"])
        self.assertNotEqual(trials[0]["id"], trials[4]["id"])

    def test_argument_bundles_booleans_and_spaces_remain_literal_tokens(self) -> None:
        self.recipe["inference"]["configurations"] = [{"parameters": {
            "--ctx-size": 4096,
            "--control-vector-layer-range": {"args": ["1", "3"]},
            "--no-warmup": True,
            "--warmup": False,
            "--custom-value": "one token; $(not a command)",
        }}]
        trial = self.plan()["trials"][0]
        self.assertEqual(trial["server_args"], [
            "--control-vector-layer-range", "1", "3", "--ctx-size", "4096",
            "--custom-value", "one token; $(not a command)", "--no-warmup",
        ])
        self.assertIs(trial["parameters"]["--warmup"], False)
        self.assertEqual(trial["parameters"], self.recipe["inference"]["configurations"][0]["parameters"])

    def test_paths_shared_overrides_and_frozen_manifest_are_recipe_relative(self) -> None:
        self.recipe["output_dir"] = "../outputs"
        self.recipe["generation"] = {"thinking": False, "seed": 777, "max_tokens": 31}
        self.recipe["execution"] = {"concurrency": 2, "startup_timeout": 0.5}
        self.recipe["inference"]["configurations"][0]["parameters"]["--ctx-size"] = [4096, 8192]
        plan = self.plan()
        self.assertEqual(plan["output_dir"], str(self.root / "outputs"))
        self.assertEqual(plan["source"], str(self.recipe_dir / "experiment.yaml"))
        options = plan["benchmark_options"]["bigcodebench"]
        self.assertEqual(options["dataset_path"], str(self.dataset))
        expected = resolve_config({"generation": self.recipe["generation"], "execution": {"concurrency": 2}})
        self.assertEqual(plan["generation"], expected["generation"])
        self.assertEqual(plan["execution"], expected["execution"] | {"startup_timeout": 0.5})
        self.assertEqual(plan["manifests"], {"bigcodebench": definition.prepare(options)[1]})
        self.assertEqual(plan["manifests"]["bigcodebench"]["count"], 3)
        self.recipe["generation"] = {"thinking": True, "seed": -42, "max_tokens": 200}
        self.recipe["inference"]["configurations"][0]["parameters"] = {"--ctx-size": 32768}
        self.assertEqual(self.plan()["manifests"], plan["manifests"])
        self.assertEqual(json.loads(json.dumps(plan, allow_nan=False)), plan)

    def test_unselected_benchmark_inputs_are_not_prepared(self) -> None:
        other = ModuleType("test_recipe.other.definition")
        other.DEFAULT_OPTIONS = copy.deepcopy(definition.DEFAULT_OPTIONS)
        other.DEFAULT_OPTIONS["dataset_path"] = "missing-unselected-dataset"
        other.PATH_OPTIONS = definition.PATH_OPTIONS
        other.validate_options = definition.validate_options
        other.prepare = definition.prepare
        self.recipe["benchmarks"] = ["bigcodebench"]
        with (
            patch.dict(registry.BENCHMARKS, {"other": "test_recipe.other"}),
            patch.dict(sys.modules, {other.__name__: other}),
        ):
            plan = self.plan()
        self.assertEqual(plan["benchmarks"], ["bigcodebench"])
        self.assertEqual(list(plan["manifests"]), ["bigcodebench"])
        self.assertEqual(
            plan["benchmark_options"]["other"]["dataset_path"],
            str(self.recipe_dir / "missing-unselected-dataset"),
        )

    def test_model_only_mode_matches_recipe_without_injecting_native_defaults(self) -> None:
        with patch.dict(definition.DEFAULT_OPTIONS, {"dataset_path": str(self.dataset), "selection": {"count": 3, "seed": 17}}):
            cli = build_plan(models=["first model.gguf", "second.gguf"], cwd=self.recipe_dir)
            self.recipe["models"] = ["first model.gguf", "second.gguf"]
            self.recipe["inference"]["configurations"][0]["parameters"] = {"--ctx-size": 8192}
            self.recipe.pop("benchmark_options")
            from_recipe = self.plan()
        for field in ("trials", "benchmarks", "benchmark_options", "generation", "execution", "manifests", "output_dir"):
            self.assertEqual(cli[field], from_recipe[field], field)
        self.assertEqual(cli["source"], "cli")
        self.assertEqual(cli["trials"][0]["server_args"], ["--ctx-size", "8192"])

    def test_yaml_duplicate_documents_unsafe_and_non_json_values_are_rejected(self) -> None:
        valid = yaml.safe_dump(self.recipe)
        invalid = [
            valid + "schema_version: 1\n",
            valid + "---\n{}\n",
            valid + "name: !!python/object/apply:os.system ['must-not-run']\n",
            valid + "name: 2026-11-03\n",
            valid + "generation: {temperature: .nan}\n",
            valid + "generation: {temperature: .inf}\n",
            valid + "generation: {seed: 1, seed: 2}\n",
            valid + "generation: {1: 2}\n",
            valid + "generation: &recursive {seed: *recursive}\n",
            valid + "generation: {seed: &recursive [*recursive]}\n",
            valid + "generation: &recursive {<<: *recursive}\n",
            "models: [\n",
        ]
        path = self.recipe_dir / "invalid.yaml"
        for text in invalid:
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_plan(recipe_path=path)

    def test_nonrecursive_aliases_can_share_atomic_argument_bundles(self) -> None:
        path = self.recipe_dir / "alias.yaml"
        path.write_text(
            "schema_version: 1\nmodels: ['first model.gguf']\n"
            "inference:\n  configurations:\n    - parameters:\n"
            "        --ctx-size: 4096\n        --first: &args {args: ['1', '3']}\n"
            "        --second: *args\n"
            "benchmark_options:\n  bigcodebench:\n    dataset_path: dataset\n    selection: {count: 3}\n",
            encoding="utf-8",
        )
        self.assertEqual(build_plan(recipe_path=path)["trials"][0]["server_args"], [
            "--ctx-size", "4096", "--first", "1", "3", "--second", "1", "3"
        ])

    def test_schema_conflicts_and_benchmark_protocol_overrides_fail(self) -> None:
        invalid_groups = [
            {"parameters": {}},
            {"parameters": {"--ctx-size": []}},
            {"parameters": {"--ctx-size": [4096, 4096]}},
            {"parameters": {"--ctx-size": True}},
            {"parameters": {"--ctx-size": 4096, "--flash-attn": True}},
            {"parameters": {"--ctx-size": 4096, "--model": "other.gguf"}},
            {"parameters": {"--ctx-size": 4096, "--temp": 1}},
            {"parameters": {"--ctx-size": 4096}, "kv_quantization": "q8"},
            {"parameters": {"--ctx-size": 4096, "--cache-type-k": "f16"}, "kv_quantization": "q8_0"},
        ]
        for group in invalid_groups:
            with self.subTest(group=group):
                recipe = copy.deepcopy(self.recipe)
                recipe["inference"]["configurations"] = [group]
                with self.assertRaises(ValueError):
                    self.plan(recipe)
        for change in (
            {"benchmarks": []}, {"benchmarks": ["unknown"]},
            {"generation": {"seed": [1, 2]}},
            {"benchmark_options": {"bigcodebench": {"dataset_path": "dataset", "selection": {"count": 9}}}},
            {"benchmark_options": {"bigcodebench": {"dataset_path": "dataset", "selection": {"count": 0}}}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan(self.recipe | change)

    def test_modes_and_missing_inputs_fail_before_trial_execution(self) -> None:
        for arguments in ({}, {"models": []}, {"models": ["first model.gguf"], "recipe_path": Path("x.yaml")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_plan(cwd=self.recipe_dir, **arguments)
        self.recipe["models"] = ["missing.gguf"]
        with self.assertRaises(FileNotFoundError):
            self.plan()
        self.recipe["models"] = ["first model.gguf"]
        self.recipe["benchmark_options"]["bigcodebench"]["dataset_path"] = "missing-dataset"
        with self.assertRaises(FileNotFoundError):
            self.plan()


if __name__ == "__main__":
    unittest.main()
