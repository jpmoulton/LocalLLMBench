"""Vendored RULER task generators and scorer (offline, pure Python, no upstream harness).

Upstream: NVIDIA RULER, Apache-2.0, github.com/NVIDIA/RULER, **verified against commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``** (fetched 2026-09-18; the pinned reference text lives
in ``tests/data/ruler-upstream-*`` and is asserted by ``tests/test_benchmarks_ruler.py``). Each
module names the upstream source file it mirrors and lists its NAMED DIVERGENCES in its own
docstring; read those before comparing any number produced here with a published RULER result, and
read ``docs/benchmarks.md`` for which comparisons are legitimate.

Two things the first, network-less vendoring got wrong and this revision corrected: upstream
appends its ``answer_prefix`` to the prompt for **every** model type (not only base models), and
``cwe`` is a **one-shot** task over a pool of single words (not a zero-shot task over
adjective-noun compounds).

Why vendored rather than installed: upstream's runner is a Docker+bash pipeline whose
``OpenAIClient`` constructs ``OpenAI(api_key=...)`` with no base URL, validates the model name
against a hardcoded GPT dictionary and counts tokens with tiktoken. None of that can point at a
private llama.cpp endpoint or count with the served model's tokenizer. Only the data generation
and the substring scorer are wanted, and both are small, deterministic and dependency-free.

Import is inert: no corpus is opened, no random number is drawn and no counter is called until a
generator runs.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .base import GeneratedItem, TaskSpec, TokenCounter, digest, user_messages
from .corpus import ESSAY_DIR, ESSAY_JSON, CorpusStatus, CorpusUnavailable, essay_status
from .fitting import DEFAULT_MAX_COUNTER_CALLS, Fit, LengthUnreachable, fit_units, normalize_count
from .scoring import (
    METRICS, STRING_MATCH_ALL, STRING_MATCH_PART, ItemScore, postprocess_pred, score_prediction,
    string_match_all, string_match_part,
)

TASK_SPECS: dict[str, TaskSpec] = {
    "niah_multikey_3": TaskSpec(
        "niah_multikey_3", "retrieval", STRING_MATCH_ALL, 128, (), "scripts/data/synthetic/niah.py",
        "UUID keys and UUID values in a haystack of other needles; no lexical shortcut to the answer."),
    "niah_multiquery": TaskSpec(
        "niah_multiquery", "retrieval", STRING_MATCH_ALL, 128, ("essay",), "scripts/data/synthetic/niah.py",
        "Four word keys with one 7-digit value each in an essay haystack, all four queried."),
    "vt": TaskSpec(
        "vt", "multi_hop", STRING_MATCH_ALL, 30, (), "scripts/data/synthetic/variable_tracking.py",
        "A four-hop chain of variable assignments scattered through repeated noise."),
    "cwe": TaskSpec(
        "cwe", "aggregation", STRING_MATCH_ALL, 120, (),
        "scripts/data/synthetic/common_words_extraction.py",
        "One-shot: ten words repeated 30x among words repeated 3x; counted, not retrieved."),
}
"""The deliberately small vendored subset, chosen for KV-quantization sensitivity."""

DEFERRED_TASKS: dict[str, str] = {
    "niah_single_1": "not vendored: a words/numbers single needle is the easiest variant and saturates",
    "niah_single_2": "not vendored: saturates; niah_multikey_3 carries the retrieval family here",
    "niah_single_3": "not vendored: saturates; niah_multikey_3 carries the retrieval family here",
    "niah_multikey_1": "not vendored: niah_multikey_3 is the harder member of the same pair",
    "niah_multikey_2": "not vendored: niah_multikey_3 is the harder member of the same pair",
    "niah_multivalue": "not vendored: niah_multiquery covers multi-string partial credit",
    "fwe": "not vendored: cwe covers the aggregation family. Its upstream parameterisation IS now "
           "verified (coded_wordlen=6, vocab_size=max_seq_length//50, counts "
           "num_words * k**-2.0 / zeta(2.0) truncated to int, vocab[0] replaced by '...', answer "
           "vocab[1:4], tokens_to_generate=50) - it needs a zeta() this image has no scipy for, and "
           "implementing a fifth task was out of scope for the verification pass",
    "qa_1": "not vendored: needs the SQuAD corpus, which this image does not bake",
    "qa_2": "not vendored: needs the HotpotQA corpus, which this image does not bake",
}
"""Upstream tasks this adapter knows by name and deliberately does not run, with the reason."""

CORPUS_STATUS: dict[str, Callable[[str], CorpusStatus]] = {"essay": essay_status}
"""Preflight probes for each corpus key a task can require."""


def task_spec(name: str) -> TaskSpec:
    spec = TASK_SPECS.get(name)
    if spec is None:
        raise KeyError(DEFERRED_TASKS.get(name, f"unknown RULER task: {name!r}"))
    return spec


def corpus_gaps(dataset_root: str, tasks: Iterable[str]) -> list[CorpusStatus]:
    """Every corpus a declared task needs and the image did not bake. Never raises."""
    needed: list[str] = []
    for name in tasks:
        for key in TASK_SPECS.get(name, TaskSpec(name, "", "", 0, (), "", "")).requires:
            if key not in needed:
                needed.append(key)
    gaps = []
    for key in needed:
        probe = CORPUS_STATUS.get(key)
        status = probe(dataset_root) if probe else CorpusStatus(key, False, None, f"no probe for corpus {key!r}")
        if not status.present:
            gaps.append(status)
    return gaps


def generate_task(name: str, *, target_tokens: int, counter: TokenCounter, seed: int,
                  dataset_root: str | None = None,
                  max_counter_calls: int = DEFAULT_MAX_COUNTER_CALLS, **options: Any) -> GeneratedItem:
    """Generate one RULER item at a requested token length using the caller's counting route."""
    spec = task_spec(name)
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if name.startswith("niah_"):
        from .niah import generate as run
    elif name == "vt":
        from .variable_tracking import generate as run
    elif name == "cwe":
        from .common_words import generate as run
    else:  # pragma: no cover - TASK_SPECS and this dispatch are kept in step by test_task_dispatch
        raise KeyError(f"no generator wired for RULER task {name!r}")
    item = run(name, target_tokens=target_tokens, counter=counter, seed=seed, dataset_root=dataset_root,
               max_counter_calls=max_counter_calls, **options)
    if item.metric != spec.metric or item.task != name or not item.answers:
        raise ValueError(f"the {name} generator returned an item that does not match its declared spec")
    return item


__all__ = [
    "CORPUS_STATUS", "DEFAULT_MAX_COUNTER_CALLS", "DEFERRED_TASKS", "ESSAY_DIR", "ESSAY_JSON", "METRICS",
    "STRING_MATCH_ALL", "STRING_MATCH_PART", "TASK_SPECS", "CorpusStatus", "CorpusUnavailable", "Fit",
    "GeneratedItem", "ItemScore", "LengthUnreachable", "TaskSpec", "TokenCounter", "corpus_gaps", "digest",
    "essay_status", "fit_units", "generate_task", "normalize_count", "postprocess_pred", "score_prediction",
    "string_match_all", "string_match_part", "task_spec", "user_messages",
]
