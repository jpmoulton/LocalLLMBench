"""Controller executor that runs one registered container candidate per manifest under a wall-clock deadline.

The executor never selects winners: it maps a manifest back to the ``ContainerRunConfig`` registered for its
configuration key, runs the injected ``ContainerRunner`` with the shared GPU lease already held by the campaign
lock, reads ``evaluation.json`` through the hash-checked artifact index and returns controller-compatible
evidence. Every failure is an ordinary failed attempt; cleanup uncertainty and the session deadline abort.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from ..analysis import config_key
from ..config import CampaignPolicy, ExperimentManifest, TaskSelection, canonical_json
from .config import BenchmarkSelection, ContainerRunConfig, NAME
from .gguf import template_hash

INDEX_NAME = "artifact-index.json"
COPIED_ARTIFACTS = ("result.json", "settings-evidence.json", "stages.jsonl")
CANDIDATE_RECORD = "candidate.json"


class SessionDeadlineExceeded(RuntimeError):
    abort_campaign = True


class CampaignAbort(RuntimeError):
    abort_campaign = True


class BenchmarkOptionsNotCarried(CampaignAbort):
    """A benchmark selection's options did not survive the session -> manifest -> candidate round-trip.

    LIVE-013: a session declaring ``ruler_output_tokens: 512`` produced candidates that ran with upstream
    RULER's per-task caps instead, so every ``vt`` answer was truncated at 30 tokens and the truncation was
    scored as a retrieval failure. A difference between what the session asked for and what the candidate
    runs is never recoverable after the fact, so it aborts the campaign instead of measuring quietly.
    """


# Error-event type names the controller records for exceptions that carry abort_campaign=True.
ABORTING_ERROR_TYPES = frozenset({SessionDeadlineExceeded.__name__, CampaignAbort.__name__,
                                  BenchmarkOptionsNotCarried.__name__})


def campaign_identity(manifests, policy: CampaignPolicy, holdout_selections) -> str:
    """Identical to controller.run_campaign's campaign_id (asserted by tests); the ledger stores it."""
    return hashlib.sha256(canonical_json({"policy": policy.model_dump(mode="json"),
                                          "manifests": [item.fingerprint() for item in manifests],
                                          "holdout": [item.model_dump(mode="json") for item in holdout_selections]
                                          }).encode()).hexdigest()


def read_indexed_artifact(run_dir: Path, name: str, *, max_bytes: int) -> bytes:
    """Bytes of one artifact named in ``artifact-index.json``; size and sha256 must match the index."""
    index_path = run_dir / INDEX_NAME
    if not index_path.is_file():
        raise ValueError(f"{INDEX_NAME} is missing from the run directory")
    if index_path.stat().st_size > max_bytes:
        raise ValueError(f"{INDEX_NAME} exceeds {max_bytes} bytes")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("artifacts") if isinstance(index, dict) and index.get("schema_version") == 1 else None
    if not isinstance(entries, list):
        raise ValueError(f"{INDEX_NAME} is not a schema 1 artifact index")
    entry = next((row for row in entries if isinstance(row, dict) and row.get("path") == name), None)
    if entry is None:
        raise ValueError(f"{name} is not in the artifact index")
    size, digest = entry.get("size"), entry.get("sha256")
    if type(size) is not int or size < 0 or size > max_bytes or not isinstance(digest, str):
        raise ValueError(f"{name} index entry is malformed or exceeds {max_bytes} bytes")
    target = run_dir / Path(*name.split("/"))
    if not target.is_file():
        raise ValueError(f"{name} is missing from the run directory")
    with target.open("rb") as handle:
        content = handle.read(size + 1)
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{name} does not match its artifact index entry")
    return content


def _benchmarks_from(tasks: tuple[TaskSelection, ...]) -> tuple[BenchmarkSelection, ...]:
    """Container benchmark selections rebuilt from the manifest's task selections.

    Options are measurement-affecting and must be restored exactly; ``_assert_options_carried`` checks the
    rebuilt configuration afterwards, so a loss anywhere in the trip refuses instead of running. A manifest
    option the container's own validator rejects (the two validators drifting apart) is a named refusal here
    rather than a bare ValidationError, because the campaign must say what it could not carry.
    """
    rebuilt = []
    for item in tasks:
        try:
            rebuilt.append(BenchmarkSelection(benchmark_id=item.suite, revision=item.revision,
                                              task_ids=item.task_ids, split=item.split, seed=item.fixture_seed,
                                              options=dict(item.options)))
        except ValueError as exc:
            raise BenchmarkOptionsNotCarried(
                f"benchmark {item.suite} declares options the candidate cannot accept, so it cannot be shown to "
                f"run what the session asked for: {type(exc).__name__}: {exc}") from exc
    return tuple(rebuilt)


def _assert_options_carried(tasks: tuple[TaskSelection, ...], selections) -> None:
    """Refuse a candidate whose benchmark options differ from the ones the manifest declares.

    This is the belt-and-braces guard for LIVE-013: the defect was not that one field was missing, it was
    that the candidate ran with settings the session never asked for and said nothing. Any future field that
    will not survive the rebuild lands here as a named abort with both option sets quoted.
    """
    if len(tasks) != len(selections):
        raise BenchmarkOptionsNotCarried(f"the manifest declares {len(tasks)} task selections but the candidate "
                                         f"rebuilt {len(selections)}; the two cannot be compared")
    for task, selection in zip(tasks, selections):
        try:
            declared, carried = canonical_json(task.options), canonical_json(selection.options)
        except (TypeError, ValueError) as exc:
            raise BenchmarkOptionsNotCarried(
                f"benchmark {task.suite} declares options that cannot be serialised, so the candidate cannot be "
                f"shown to run what the session asked for: {type(exc).__name__}: {exc}") from exc
        if declared != carried:
            raise BenchmarkOptionsNotCarried(
                f"benchmark {task.suite} would run with options {carried} but the manifest declares {declared}; "
                "refusing to measure a configuration the session did not ask for")


class ContainerExecutor:
    def __init__(self, session, bundle, *, runner, output_root: str | Path, deadline_epoch: float,
                 template_hashes: dict[str, str], clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.session, self.bundle, self.runner = session, bundle, runner
        self.output_root = Path(output_root)
        self.deadline_epoch, self.template_hashes = float(deadline_epoch), dict(template_hashes)
        self.clock, self.wall = clock, wall
        self._configs: dict[str, ContainerRunConfig] = {}
        self._artifact_sink = self._sample_sink = self._event_sink = None
        self.deadline_hits = 0
        self.runs: list[dict[str, Any]] = []
        candidates = self.output_root / "candidates"
        existing = [int(match[1]) for path in candidates.iterdir() if candidates.is_dir()
                    for match in [re.fullmatch(r"(\d{3})-.*", path.name)] if match] if candidates.is_dir() else []
        self._next = max(existing, default=-1) + 1

    def register(self, config: ContainerRunConfig, manifest: ExperimentManifest) -> None:
        key = config_key(manifest)
        if key in self._configs and self._configs[key].fingerprint() != config.fingerprint():
            raise ValueError("configuration key already registered with a different container config")
        self._configs[key] = config

    def bind_artifact_sink(self, sink) -> None:
        self._artifact_sink = sink

    def bind_sample_sink(self, sink) -> None:
        self._sample_sink = sink

    def bind_event_sink(self, sink) -> None:
        self._event_sink = sink

    def config_for(self, manifest: ExperimentManifest) -> ContainerRunConfig:
        registered = self._configs.get(config_key(manifest))
        if registered is None:
            raise ValueError("manifest does not correspond to a registered container configuration")
        raw = registered.model_dump(mode="json")
        raw["benchmarks"] = [item.model_dump(mode="json") for item in _benchmarks_from(manifest.tasks)]
        if any(item.split == "holdout" for item in manifest.tasks):
            raw["label"] = (registered.label[:55] + "-holdout")
            if not re.fullmatch(NAME, raw["label"]):
                raise ValueError("holdout label is not a safe name")
        rebuilt = ContainerRunConfig.model_validate_json(canonical_json(raw))
        _assert_options_carried(manifest.tasks, rebuilt.benchmarks)
        return rebuilt

    def _emit(self, event: dict) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    def __call__(self, manifest: ExperimentManifest, *, timeout_seconds: float) -> dict:
        config = self.config_for(manifest)
        remaining = self.deadline_epoch - self.wall()
        floor = config.bounds.minimum_wall_seconds() + config.bounds.cleanup_reserve_seconds
        if remaining < floor:
            self.deadline_hits += 1
            self._emit({"event": "session-deadline", "remaining_seconds": remaining, "required_seconds": floor})
            raise SessionDeadlineExceeded(f"session deadline: {remaining:.0f}s remain, a candidate needs {floor}s")
        index = self._next
        self._next += 1
        run_dir = self.output_root / "candidates" / f"{index:03d}-{config.label}"
        budget = float(min(timeout_seconds, remaining))
        self._emit({"event": "candidate-start", "label": config.label, "run_dir": str(run_dir),
                    "budget_seconds": budget, "config_fingerprint": config.fingerprint()})
        record = {"label": config.label, "run_dir": str(run_dir), "config_fingerprint": config.fingerprint(),
                  "budget_seconds": budget, "state": None, "attempt_id": None, "failure_stage": None,
                  "failure_reasons": []}
        try:
            result = self.runner.run(config, run_dir, remaining_budget_seconds=budget, lease_held=True)
        except BaseException as exc:
            record.update(state="exception", error=f"{type(exc).__name__}: {exc}")
            self.runs.append(record)
            try:
                self._store_record(record)  # the run directory stays discoverable even for an interrupted candidate
            except Exception as store_exc:  # never mask the runner's exception with a Store failure
                self._emit({"event": "candidate-record-not-stored", "label": config.label,
                            "reason": f"{type(store_exc).__name__}: {store_exc}"})
            self._emit({"event": "candidate-end", **record})
            raise
        record.update(state=result.state, attempt_id=result.attempt_id, failure_stage=result.failure_stage,
                      failure_reasons=list(result.failure_reasons))
        self.runs.append(record)
        self._store_record(record)
        self._emit({"event": "candidate-end", **record})
        max_bytes = config.limits.max_artifact_bytes
        not_copied = self._copy_artifacts(run_dir, max_bytes)
        if result.abort_campaign:
            raise CampaignAbort(f"candidate {config.label} left cleanup uncertain: "
                                + "; ".join(list(result.failure_reasons) + not_copied))
        if result.state != "completed":
            raise ValueError(f"candidate {config.label} {result.state} at stage {result.failure_stage}: "
                             + "; ".join(list(result.failure_reasons) + not_copied))
        evaluation = json.loads(read_indexed_artifact(run_dir, "evaluation.json", max_bytes=max_bytes))
        if (not isinstance(evaluation, dict) or not isinstance(evaluation.get("samples"), list)
                or not isinstance(evaluation.get("speed_observations"), list)):
            raise ValueError("evaluation.json lacks samples or speed observations")
        from ..container_eval import speed_observations_from
        check = self._template_check(config, evaluation)
        dumped = result.model_dump(mode="json")
        return {"synthetic": result.synthetic, "model_evaluated": not result.synthetic,
                "samples": evaluation["samples"], "speed_observations": speed_observations_from(evaluation),
                "effective_settings_verified": result.effective_settings_verified,
                "actual_context_verified": result.actual_context_verified and check["matches"],
                "holdout_passed": False, "comparisons": {}, "samples_persisted": False,
                "raw": {"result": dumped, "run_dir": str(run_dir), "label": config.label,
                        "config_fingerprint": config.fingerprint(), "template_check": check,
                        "artifact_index": dumped.get("artifacts", []),
                        "candidate_report": (dumped.get("reports") or {}).get("candidate"),
                        "warnings": ([] if check["matches"] else ["template_hash_mismatch: served chat template "
                                                                     "differs from the GGUF header template"])
                        + not_copied}}

    def _template_check(self, config: ContainerRunConfig, evaluation: dict) -> dict:
        expected = self.template_hashes.get(config.model.sha256)
        props = (evaluation.get("readback") or {}).get("props") if isinstance(evaluation.get("readback"), dict) else None
        served = props.get("chat_template") if isinstance(props, dict) else None
        observed = template_hash(served) if isinstance(served, str) and served else None
        return {"expected": expected, "observed": observed,
                "matches": expected is not None and observed is not None and expected == observed}

    def _store_record(self, record: dict) -> None:
        """candidate.json in the Store: label, run directory and container state for every attempt, failed or not."""
        if self._artifact_sink is not None:
            self._artifact_sink(CANDIDATE_RECORD, canonical_json(record).encode())

    def _copy_artifacts(self, run_dir: Path, max_bytes: int) -> list[str]:
        """Copy the indexed run artifacts into the Store; returns one ``artifact_unreadable:<name>: <why>`` per
        artifact that could not be copied so the report row says why a column is missing."""
        warnings = []
        if self._artifact_sink is None:
            return warnings
        for name in COPIED_ARTIFACTS:
            try:
                content = read_indexed_artifact(run_dir, name, max_bytes=max_bytes)
            except ValueError as exc:
                self._emit({"event": "artifact-not-copied", "name": name, "reason": str(exc)})
                warnings.append(f"artifact_unreadable:{name}: {exc}")
                continue
            self._artifact_sink(name, content)
        return warnings
