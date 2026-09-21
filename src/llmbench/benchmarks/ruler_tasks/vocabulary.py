"""The two word pools RULER draws from, reproduced exactly.

Mirrors NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER) at commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``. Upstream builds **two different** pools and the
previous revision of this file conflated them:

* ``scripts/data/synthetic/niah.py`` (needle keys/values of type ``words``)::

      nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
      adjs  = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
      words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
      words = sorted(list(set(words)))          # 910 x 6782 = 6_171_620 compounds
      ... random.choice(words)

* ``scripts/data/synthetic/common_words_extraction.py`` (the ``cwe`` list)::

      words = nouns + adjs + verbs
      words = sorted(list(set(words)))          # 8166 single words
      random.Random(args.random_seed).shuffle(words)
      ... random.sample(words, num_words)

  and, only when ``num_words > len(words)``, a fallback pool read from
  ``scripts/data/synthetic/json/english_words.json``.

Both wonderwords lists are vendored verbatim in ``wonderwords_lists.py`` with their upstream
SHA-256s, so the pools here are upstream's pools, not an approximation of them.

The 6.17M compound list is never materialised: ``sorted(set(...))`` of ``f"{adj}-{noun}"`` is
exactly ``sorted(set(adjs))`` major / ``sorted(set(nouns))`` minor, because no adjective is a
proper prefix of another adjective whose next character sorts below ``"-"`` (verified in the test
suite, and re-checked here by :func:`_assert_factorisation` on first use). So the k-th element is
computed in O(1) instead of building a ~600 MB list the way upstream does.

NAMED DIVERGENCE - ``cwe`` beyond 8166 distinct words. Upstream's fallback
``json/english_words.json`` is an 8.5 MB git-LFS blob (sha256
``affcd6d4...bc2eca``, 8_564_991 bytes) that this image does not bake and that cannot be
reconstructed. When ``cwe`` needs more than 8166 distinct words, upstream silently switches to that
dictionary and we instead stop growing the list and report ``units_exhausted``: the generator never
repeats a word to pad, because an extra repetition changes which words are "most common" and would
corrupt the answer key. On a Qwen-family tokenizer the crossover is around the 128K point (roughly
27k numbered entries), so ``cwe`` at 4K-64K is inside upstream's own pool and ``cwe`` at 128K is
where this limit bites. See ``docs/benchmarks.md``.
"""

from __future__ import annotations

from functools import lru_cache

from .wonderwords_lists import adjectives, nouns, verbs

CWE_SHUFFLE_SEED = 42
"""Upstream shuffles the deduplicated ``cwe`` pool once with its ``--random_seed`` (default 42)."""

ENGLISH_WORDS_JSON_SHA256 = "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca"
"""git-LFS oid of upstream's ``cwe`` fallback dictionary; recorded, deliberately not vendored."""
ENGLISH_WORDS_JSON_BYTES = 8_564_991


@lru_cache(maxsize=1)
def _parts() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sorted(set(adjectives()))), tuple(sorted(set(nouns())))


def _assert_factorisation(sorted_adjectives: tuple[str, ...]) -> None:
    """``sorted(set(adj + "-" + noun))`` groups by adjective only if no prefix pair inverts it."""
    for index, adjective in enumerate(sorted_adjectives):
        follower = index + 1
        while follower < len(sorted_adjectives) and sorted_adjectives[follower].startswith(adjective):
            if sorted_adjectives[follower][len(adjective)] < "-":
                raise ValueError(
                    "the wonderwords adjective list no longer factorises the RULER compound order: "
                    f"{adjective!r} before {sorted_adjectives[follower]!r}")
            follower += 1


@lru_cache(maxsize=1)
def niah_compound_count() -> int:
    """``len(sorted(list(set(words))))`` in upstream ``niah.py``: 910 x 6782 = 6_171_620."""
    sorted_adjectives, sorted_nouns = _parts()
    _assert_factorisation(sorted_adjectives)
    return len(sorted_adjectives) * len(sorted_nouns)


def niah_compound(index: int) -> str:
    """The ``index``-th entry of upstream's sorted, deduplicated ``adjective-noun`` compound list."""
    total = niah_compound_count()
    if type(index) is not int or isinstance(index, bool) or not 0 <= index < total:
        raise ValueError(f"compound index must be in [0, {total})")
    sorted_adjectives, sorted_nouns = _parts()
    stride = len(sorted_nouns)
    return f"{sorted_adjectives[index // stride]}-{sorted_nouns[index % stride]}"


@lru_cache(maxsize=1)
def cwe_word_pool() -> tuple[str, ...]:
    """Upstream's ``cwe`` pool: ``sorted(set(nouns + adjs + verbs))`` shuffled with ``Random(42)``."""
    import random

    pool = sorted(set(nouns()) | set(adjectives()) | set(verbs()))
    random.Random(CWE_SHUFFLE_SEED).shuffle(pool)
    return tuple(pool)


def cwe_pool_size() -> int:
    return len(cwe_word_pool())
