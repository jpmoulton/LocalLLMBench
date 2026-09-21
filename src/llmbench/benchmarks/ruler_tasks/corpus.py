"""Haystack corpora the image must bake, and honest absence when it did not.

Mirrors NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER) at commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``, ``scripts/data/synthetic/niah.py``::

    essay = json.load(open(essay))['text']
    haystack = re.sub(r'\\s+', " ", essay).split(" ")      # the unit is a WORD
    ...
    text = " ".join(haystack[:num_haystack])
    document_sents = sent_tokenize(text.strip())           # NLTK punkt

The dump itself is what ``scripts/data/synthetic/json/download_paulgraham_essay.py`` produces: a
JSON object with a single ``text`` field.

Nothing here downloads anything. When the file is absent the caller is told exactly which path was
expected; a generator never substitutes lorem-ipsum, repeated filler or another task's haystack
for a missing corpus, because a retrieval score over filler is not the score it claims to be.

NAMED DIVERGENCE 1 - sentence splitting. Upstream splits the word slice into sentences with NLTK's
punkt tokenizer; punkt is a trained model this image does not carry, so :func:`sentence_split` uses
the regex ``(?<=[.!?])\\s+``. (OpenCompass's independent RULER port replaces punkt with the even
coarser ``text.split('. ')``, so neither third-party port reproduces punkt.) The consequence is
bounded and local: the *unit* of length growth is still upstream's word, only the needle insertion
points differ, and a mis-detected boundary (``Mr.``, ``i.e.``, ``3.5``, a quoted period) moves a
needle by at most one sentence - on the order of 20 words inside a haystack of 10^5 words, i.e.
under 0.02% of depth. It cannot change how many needles there are, what they say or how they score.

NAMED DIVERGENCE 2 - exhaustion instead of repetition. Since 2025 upstream repeats the essay when
``num_haystack > len(haystack)``::

    repeats = (num_haystack + len(haystack) - 1) // len(haystack)
    text = " ".join((haystack * repeats)[:num_haystack])

This module refuses to repeat: it reports ``units_exhausted`` and the adapter fails the item closed
as ``length_target_unmet``. Duplicated prose changes retrieval difficulty (the same sentence now
appears at several depths), and running out of essay means the image baked too little text - a
defect to fix, not to paper over. With upstream's own dump this branch is never reached below 128K.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ESSAY_JSON = "ruler/PaulGrahamEssays.json"
"""Primary baked artefact: ``{"text": "..."}`` (or ``{"text": ["...", ...]}``), relative to dataset_root."""
ESSAY_DIR = "ruler/essays"
"""Fallback: a directory of ``*.txt`` essays, read in sorted filename order."""
MAX_CORPUS_BYTES = 64 * 1024 * 1024
MIN_CORPUS_SENTENCES = 64

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
"""Stand-in for upstream's NLTK punkt ``sent_tokenize``; see NAMED DIVERGENCE 1 above."""
_WHITESPACE = re.compile(r"\s+")


def sentence_split(text: str) -> list[str]:
    """Upstream's ``sent_tokenize(text.strip())``, approximated by a punctuation regex."""
    return [part for part in (piece.strip() for piece in _SENTENCE.split(text.strip())) if part]


class CorpusUnavailable(FileNotFoundError):
    """A haystack corpus the requested task needs is not present under ``dataset_root``."""


@dataclass(frozen=True)
class CorpusStatus:
    key: str
    present: bool
    path: str | None
    detail: str


def _candidates(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root)
    return root / ESSAY_JSON, root / ESSAY_DIR


def essay_status(dataset_root: str | Path) -> CorpusStatus:
    """Cheap presence check for preflight; never reads or parses the whole corpus, never raises."""
    try:
        json_path, dir_path = _candidates(dataset_root)
        if json_path.is_file() and json_path.stat().st_size > 0:
            return CorpusStatus("essay", True, str(json_path), "essay JSON present")
        if dir_path.is_dir() and any(dir_path.glob("*.txt")):
            return CorpusStatus("essay", True, str(dir_path), "essay text directory present")
        return CorpusStatus("essay", False, None,
                            f"the Paul Graham essay haystack is absent: expected {json_path} "
                            f"or {dir_path}/*.txt under the image's dataset root")
    except OSError as exc:
        return CorpusStatus("essay", False, None, f"the essay haystack could not be inspected: {exc}")


def _read_text(dataset_root: str | Path) -> str:
    json_path, dir_path = _candidates(dataset_root)
    if json_path.is_file():
        size = json_path.stat().st_size
        if size > MAX_CORPUS_BYTES:
            raise CorpusUnavailable(f"{json_path} is {size} bytes, above the {MAX_CORPUS_BYTES}-byte bound")
        try:
            payload = json.loads(json_path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusUnavailable(f"{json_path} is not readable UTF-8 JSON: {exc}") from exc
        text = payload.get("text") if isinstance(payload, dict) else payload
        if isinstance(text, list):
            text = "\n".join(part for part in text if isinstance(part, str))
        if not isinstance(text, str) or not text.strip():
            raise CorpusUnavailable(f"{json_path} has no non-empty 'text' field")
        return text
    if dir_path.is_dir():
        files = sorted(path for path in dir_path.glob("*.txt") if path.is_file())
        total = sum(path.stat().st_size for path in files)
        if not files:
            raise CorpusUnavailable(f"{dir_path} holds no *.txt essay")
        if total > MAX_CORPUS_BYTES:
            raise CorpusUnavailable(f"{dir_path} holds {total} bytes, above the {MAX_CORPUS_BYTES}-byte bound")
        try:
            return "\n".join(path.read_bytes().decode("utf-8", "replace") for path in files)
        except OSError as exc:
            raise CorpusUnavailable(f"{dir_path} could not be read: {exc}") from exc
    raise CorpusUnavailable(essay_status(dataset_root).detail)


@lru_cache(maxsize=4)
def essay_words(dataset_root: str) -> tuple[str, ...]:
    """Upstream's ``re.sub(r'\\s+', " ", essay).split(" ")``: the haystack unit is one word.

    The leading/trailing ``strip`` is the only addition, and only so a corpus that begins or ends
    with whitespace cannot spend a haystack unit on an empty string; upstream reaches the same
    place via ``sent_tokenize(text.strip())``.
    """
    words = tuple(_WHITESPACE.sub(" ", _read_text(dataset_root)).strip().split(" "))
    sentences = sentence_split(" ".join(words))
    if len(sentences) < MIN_CORPUS_SENTENCES:
        raise CorpusUnavailable(
            f"the essay haystack holds {len(sentences)} sentences, below the {MIN_CORPUS_SENTENCES} "
            "needed to build a long-context item")
    return words


def essay_sentences(dataset_root: str) -> tuple[str, ...]:
    """The same corpus seen as sentences - preflight and tests only; the generator slices words."""
    return tuple(sentence_split(" ".join(essay_words(dataset_root))))
