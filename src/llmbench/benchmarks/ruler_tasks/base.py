"""Shared types for the vendored RULER generators.

The generators mirror NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER); this module only holds
the shapes they return and the helpers they share. Import is inert: no corpus is read, no counter
is called, no randomness is drawn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

TokenCounter = Callable[[list[dict[str, str]], list[dict[str, Any]]], Any]
"""The suite's counting route: ``(messages, tools) -> int`` or ``{"tokens": int, ...}``."""


@dataclass(frozen=True)
class TaskSpec:
    """One declared RULER task: what it is, what it needs, and how upstream scores it."""

    name: str
    family: str
    metric: str
    max_output_tokens: int
    """Upstream ``tokens_to_generate`` for this task family (constants.py)."""
    requires: tuple[str, ...]
    """Corpus keys that must be baked into the image for this task to run at all."""
    upstream: str
    """The upstream source file this generator mirrors."""
    description: str


@dataclass(frozen=True)
class GeneratedItem:
    """A generated prompt with its reference answers and the evidence of how long it really is."""

    task: str
    seed: int
    target_tokens: int
    actual_tokens: int
    units: int
    counter_calls: int
    messages: tuple[dict[str, str], ...]
    answers: tuple[str, ...]
    metric: str
    max_output_tokens: int
    prompt_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return "\n".join(message["content"] for message in self.messages)

    def summary(self, *, preview: int = 240) -> dict[str, Any]:
        """Persistable record: hashes and counts, never the whole 128K-token prompt."""
        text = self.prompt
        return {"task": self.task, "seed": self.seed, "target_tokens": self.target_tokens,
                "actual_tokens": self.actual_tokens, "units": self.units,
                "counter_calls": self.counter_calls, "metric": self.metric,
                "max_output_tokens": self.max_output_tokens, "prompt_sha256": self.prompt_sha256,
                "prompt_characters": len(text), "prompt_head": text[:preview], "prompt_tail": text[-preview:],
                "answers": list(self.answers), "metadata": dict(self.metadata)}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def user_messages(prompt: str) -> tuple[dict[str, str], ...]:
    """One user turn, no system prompt. The caller has already appended upstream's answer prefix.

    NAMED DIVERGENCE, verified against upstream at commit
    ``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``. ``scripts/data/prepare.py`` builds
    ``model_template.format(task_template=task_template) + answer_prefix`` for **every** model type,
    not only base models, and ``scripts/pred/call_api.py`` then sends
    ``data_point['input'] + data_point.get('answer_prefix', '')``. For a chat model the "model
    template" is a literal string from ``scripts/data/template.py`` (``meta-llama3``, ``Phi3``, ...),
    so the prefix lands *after* the assistant marker: upstream prefills the assistant turn with it.

    A chat-completions endpoint cannot prefill an assistant turn, so the generators put the same
    prefix bytes at the end of the single user turn. Every generator records
    ``metadata["answer_prefix_in_prompt"] = True`` so a row can never be read as if the prefix were
    absent. (An earlier revision of this file asserted that upstream omits the prefix on the chat
    path. That was wrong; it was written without network access and is corrected here.)
    """
    return ({"role": "user", "content": prompt},)


def count_messages(counter: TokenCounter, messages: Sequence[Mapping[str, str]]) -> Any:
    return counter([dict(message) for message in messages], [])
