"""Small original coding fixtures; evaluator answers never enter model workspaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..config import canonical_json


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str


@dataclass(frozen=True)
class CheckCase:
    case_id: str
    arguments_json: str
    expected_json: str


@dataclass(frozen=True)
class CodingFixture:
    fixture_id: str
    language: str
    prompt: str
    entrypoint: str
    initial_files: tuple[SourceFile, ...]
    reference_files: tuple[SourceFile, ...]
    checks: tuple[CheckCase, ...]
    revision: str = "private-coding-v1"

    def identity(self) -> str:
        from dataclasses import asdict
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()


def fixtures() -> tuple[CodingFixture, ...]:
    """Fresh immutable data. Reference patches are evaluator-only calibration data."""
    return (
        CodingFixture(
            fixture_id="python/chunks", language="python", entrypoint="batches:chunked",
            prompt="Implement chunked(values, size). Return consecutive list chunks in input order, "
                   "including the final short chunk. Do not mutate values. Empty input returns []. "
                   "Raise ValueError for size <= 0. Modify only batches.py.",
            initial_files=(SourceFile("batches.py", "def chunked(values, size):\n    return [values]\n"),),
            reference_files=(SourceFile("batches.py", "def chunked(values, size):\n"
                                        "    if size <= 0:\n        raise ValueError('size must be positive')\n"
                                        "    return [values[i:i + size] for i in range(0, len(values), size)]\n"),),
            checks=(CheckCase("tail", '[[1,2,3,4,5],2]', '[[1,2],[3,4],[5]]'),
                    CheckCase("empty", '[[],3]', '[]'),
                    CheckCase("wide", '[[1,2],5]', '[[1,2]]'),
                    CheckCase("bad-size", '[[1],0]', '{"error":"ValueError"}')),
        ),
        CodingFixture(
            fixture_id="typescript/group-records", language="typescript", entrypoint="group:groupRecords",
            prompt="Implement exported groupRecords(records) in group.ts. Each record has a string key "
                   "and numeric value. Return an array of {key, values:number[]} groups in first-key "
                   "appearance order; preserve value order and duplicates. Treat keys literally, "
                   "including __proto__. Do not mutate input. Use TypeScript types and compile strictly.",
            initial_files=(SourceFile("group.ts", "export type RecordItem = { key: string; value: number };\n"
                                      "export function groupRecords(records: RecordItem[]): "
                                      "{key:string; values:number[]}[] { return []; }\n"),),
            reference_files=(SourceFile("group.ts", "export type RecordItem = { key: string; value: number };\n"
                                        "export function groupRecords(records: RecordItem[]): "
                                        "{key:string; values:number[]}[] {\n"
                                        "  const groups = new Map<string, number[]>();\n"
                                        "  for (const {key, value} of records) {\n"
                                        "    const values = groups.get(key);\n"
                                        "    if (values) values.push(value); else groups.set(key, [value]);\n"
                                        "  }\n"
                                        "  return Array.from(groups, ([key, values]) => ({key, values}));\n}\n"),),
            checks=(CheckCase("ordering", '[[{"key":"b","value":2},{"key":"a","value":1},'
                                           '{"key":"b","value":2}]]',
                                           '[{"key":"b","values":[2,2]},{"key":"a","values":[1]}]'),
                    CheckCase("literal-key", '[[{"key":"__proto__","value":7}]]',
                              '[{"key":"__proto__","values":[7]}]'),
                    CheckCase("empty", '[[]]', '[]')),
        ),
        CodingFixture(
            fixture_id="javascript/stable-unique", language="javascript", entrypoint="unique:stableUnique",
            prompt="Implement stableUnique(values) in unique.cjs. Given strings, return each distinct "
                   "string once in first appearance order, case sensitively. Preserve empty strings "
                   "and literal __proto__. Do not mutate input. Export using module.exports.",
            initial_files=(SourceFile("unique.cjs", "exports.stableUnique = values => values;\n"),),
            reference_files=(SourceFile("unique.cjs", "exports.stableUnique = values => [...new Set(values)];\n"),),
            checks=(CheckCase("order", '[["b","a","b","A","a"]]', '["b","a","A"]'),
                    CheckCase("special", '[["","__proto__","", "__proto__"]]', '["","__proto__"]'),
                    CheckCase("empty", '[[]]', '[]')),
        ),
    )


def write_candidate(root: str | Path, fixture: CodingFixture,
                    patch: Mapping[str, str] | None = None, *, max_bytes: int = 1_048_576) -> Path:
    """Stage only model-visible sources into a new dedicated directory, never reference/tests."""
    destination = Path(root)
    if destination.exists():
        raise ValueError("Candidate directory must be new; refusing overwrite")
    allowed = {item.path for item in fixture.initial_files}
    changes = dict(patch or {})
    if set(changes) - allowed:
        raise ValueError("Patch includes a file outside this fixture's editable source list")
    sources = {item.path: changes.get(item.path, item.content) for item in fixture.initial_files}
    if max_bytes <= 0 or sum(len(value.encode("utf-8")) for value in sources.values()) > max_bytes:
        raise ValueError("Candidate source exceeds byte budget")
    for name in sources:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Invalid fixture source path")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in sources.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return destination.resolve()


@dataclass(frozen=True)
class CodingScore:
    passed: int
    required: int
    completed: int
    success: bool
    failed_case_ids: tuple[str, ...]
    label: str = "private fixture exact observations; not a public benchmark score"


def score_observations(fixture: CodingFixture, observations: Mapping[str, Any], *,
                       compile_ok: bool, execution_ok: bool) -> CodingScore:
    """Compare evaluator-owned expectations; exit code/model declarations alone never pass."""
    required_ids = {case.case_id for case in fixture.checks}
    if type(compile_ok) is not bool or type(execution_ok) is not bool:
        raise ValueError("Compilation/execution evidence must be booleans")
    if set(observations) - required_ids:
        raise ValueError("Observations include unknown check IDs")
    failed = []
    for case in fixture.checks:
        if (not compile_ok or not execution_ok or case.case_id not in observations
                or canonical_json(observations[case.case_id]) != canonical_json(json.loads(case.expected_json))):
            failed.append(case.case_id)
    return CodingScore(len(fixture.checks) - len(failed), len(fixture.checks), len(observations),
                       not failed, tuple(failed))
