"""RULER common words extraction (``cwe``).

Mirrors NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER) at commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``: ``scripts/data/synthetic/common_words_extraction.py``,
the ``common_words_extraction`` entry of ``scripts/data/synthetic/constants.py`` and the ``cwe``
entry of ``scripts/synthetic.yaml`` (``freq_cw=30``, ``freq_ucw=3``, ``num_cw=10``).

A numbered word list is built so that ten "common" words occur ``freq_cw`` times each and every
other word occurs ``freq_ucw`` times; the list is shuffled with a fixed seed and the model is asked
for the ten most common words. This is the aggregation family: the answer is not stored anywhere in
the context, it has to be counted across the whole window, which is a different failure mode from
retrieval when the KV cache is quantized.

Reproduced verbatim from upstream, **including two things the first vendoring of this file missed**:

 1. **The word pool is single words, not compounds.** ``common_words_extraction.py`` uses
    ``sorted(set(nouns + adjs + verbs))`` from ``wonderwords`` - 8166 single words - shuffled once
    with ``Random(42)``. Only ``niah.py`` builds ``adjective-noun`` compounds. See ``vocabulary.py``.
 2. **The prompt is one-shot, not zero-shot.** ``generate_input_output`` prepends
    ``num_fewshot=1`` fully worked example - a short list plus its numbered answer - and joins it to
    the real question with ``"\\n"``. A zero-shot ``cwe`` asks a materially different question,
    because the answer format that ``string_match_all`` is looking for is demonstrated only by the
    shot.

Also reproduced: the ``"{i}. {word}"`` numbering joined by single spaces, the fixed-seed
(``Random(42)``) shuffle of the assembled list, the ``num_words <= 4096``-style short-context
branch (``6``/``1`` repeats and a 20-word shot instead of ``30``/``3`` and a 40-word shot), the
template and its ``answer_prefix``.

NAMED DIVERGENCES:

 1. **Answer prefix placement** - as in ``niah.py``: upstream's ``answer_prefix`` is appended after
    the chat template's assistant marker (an assistant prefill); here the identical bytes end the
    single user turn, and the shot is rendered with the prefix exactly as upstream renders it.
 2. **Pool exhaustion instead of a fallback dictionary** - when ``num_words > 8166`` upstream
    switches to ``json/english_words.json``, an 8.5 MB git-LFS blob this image does not bake. Here
    the list simply stops growing and the item reports ``units_exhausted``; the adapter then fails
    it closed as ``length_target_unmet``. The generator never repeats a word to pad, because an
    extra repetition changes which words are "most common" and would silently corrupt the answer
    key. On a Qwen-family tokenizer this bites at roughly the 128K point and not below.
 3. **Stable render** - the ten common words are the first ten of one permutation of the pool drawn
    once per item, and growing the list only takes more of the same permutation, so the answer key
    does not change while the length search runs. Upstream re-runs ``random.sample(words,
    num_words)`` at every candidate size, which redraws the common words too. One permutation
    prefix and ``random.sample`` are the same distribution; only the per-probe redraw is dropped.
    The shot is drawn from its own independent sample of the same pool, as upstream's is.
"""

from __future__ import annotations

import random
from typing import Any

from .base import GeneratedItem, TokenCounter, count_messages, digest, user_messages
from .fitting import DEFAULT_MAX_COUNTER_CALLS, fit_units
from .scoring import STRING_MATCH_ALL
from .vocabulary import cwe_word_pool

TEMPLATE = ("Below is a numbered list of words. In these words, some appear more often than others. "
            "Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common "
            "words in the above list?")
"""``TASKS['common_words_extraction']['template']`` in upstream ``constants.py``."""
ANSWER_PREFIX = " Answer: The top 10 words that appear most often in the list are:"
"""``TASKS['common_words_extraction']['answer_prefix']``; ``prepare.py`` concatenates it on."""
MAX_OUTPUT_TOKENS = 120
"""Upstream ``tokens_to_generate`` for common words extraction."""
FREQ_CW = 30
FREQ_UCW = 3
NUM_CW = 10
NUM_FEWSHOT = 1
"""Upstream ``--num_fewshot`` default."""
SHUFFLE_SEED = 42
"""Upstream shuffles the assembled word list with its fixed ``random_seed`` (42 by default)."""
SHORT_CONTEXT_TOKENS = 4096
"""Upstream switches to the easier repeat counts when ``max_seq_length < 4096``."""
SHORT_BRANCH = {"shot_words": 20, "shot_cw": 3, "shot_ucw": 1, "freq_cw": 6, "freq_ucw": 1}
LONG_BRANCH = {"shot_words": 40, "shot_cw": 10, "shot_ucw": 3, "freq_cw": FREQ_CW, "freq_ucw": FREQ_UCW}


def numbered(words: list[str]) -> str:
    """Upstream's ``' '.join([f"{i + 1}. {word}" for i, word in enumerate(word_list)])``."""
    return " ".join(f"{index + 1}. {word}" for index, word in enumerate(words))


def build_list(common: list[str], uncommon: list[str], common_repeats: int, uncommon_repeats: int) -> str:
    """Upstream's ``get_example`` body from the split onwards, including the ``Random(42)`` shuffle."""
    words = list(common) * int(common_repeats) + list(uncommon) * int(uncommon_repeats)
    random.Random(SHUFFLE_SEED).shuffle(words)
    return numbered(words)


def generate(name: str, *, target_tokens: int, counter: TokenCounter, seed: int,
             dataset_root: str | None = None, max_counter_calls: int = DEFAULT_MAX_COUNTER_CALLS,
             freq_cw: int | None = None, freq_ucw: int | None = None, num_cw: int = NUM_CW,
             num_fewshot: int = NUM_FEWSHOT) -> GeneratedItem:
    if name != "cwe":
        raise ValueError(f"unknown aggregation task: {name!r}")
    pool = cwe_word_pool()
    # Upstream branches on ``max_seq_length``, which is the whole budget: input plus generation.
    branch = LONG_BRANCH if target_tokens + MAX_OUTPUT_TOKENS >= SHORT_CONTEXT_TOKENS else SHORT_BRANCH
    freq_cw = branch["freq_cw"] if freq_cw is None else freq_cw
    freq_ucw = branch["freq_ucw"] if freq_ucw is None else freq_ucw

    rng = random.Random(seed)
    order = rng.sample(pool, len(pool))
    common, rest = order[:num_cw], order[num_cw:]
    template = TEMPLATE + ANSWER_PREFIX

    shots = []
    for _ in range(num_fewshot):
        shot = rng.sample(pool, branch["shot_words"])
        shot_common, shot_uncommon = shot[:num_cw], shot[num_cw:]
        shots.append(template.format(context=build_list(shot_common, shot_uncommon, branch["shot_cw"],
                                                        branch["shot_ucw"]))
                     + " " + numbered(shot_common))
    preamble = "\n".join(shots) + "\n"
    max_units = len(pool)

    def render(units: int) -> tuple[dict[str, str], ...]:
        context = build_list(common, rest[:max(units - num_cw, 0)], freq_cw, freq_ucw)
        return user_messages(preamble + template.format(context=context))

    fit = fit_units(lambda units: count_messages(counter, render(units)), target_tokens=target_tokens,
                    max_units=max_units, max_counter_calls=max_counter_calls)
    messages = render(fit.units)
    uncommon_words = max(fit.units - num_cw, 0)
    metadata: dict[str, Any] = {
        "family": "aggregation", "freq_cw": freq_cw, "freq_ucw": freq_ucw, "num_cw": num_cw,
        "num_words": fit.units, "uncommon_words": uncommon_words,
        "total_words": num_cw * freq_cw + uncommon_words * freq_ucw,
        "haystack_unit": "distinct_word", "units_exhausted": fit.units >= max_units,
        "template_tokens": fit.base_tokens, "word_pool_size": len(pool),
        "num_fewshot": num_fewshot, "fewshot_words": branch["shot_words"],
        "short_context_branch": branch is SHORT_BRANCH, "answer_prefix_in_prompt": True,
    }
    return GeneratedItem("cwe", seed, target_tokens, fit.tokens, fit.units, fit.counter_calls, messages,
                         tuple(common), STRING_MATCH_ALL, MAX_OUTPUT_TOKENS, digest(list(messages)), metadata)
