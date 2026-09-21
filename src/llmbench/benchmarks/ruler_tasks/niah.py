"""RULER needle-in-a-haystack retrieval tasks.

Mirrors NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER) at commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``: ``scripts/data/synthetic/niah.py``, the ``niah``
entry of ``scripts/data/synthetic/constants.py`` and the ``niah_*`` entries of
``scripts/synthetic.yaml``. Vendored because upstream's runner is Docker+bash and its
``OpenAIClient`` hardcodes an api.openai.com client with a GPT-only model whitelist and a tiktoken
counter; only the data generation is wanted, and it must count with the served model's tokenizer.

Vendored variants (chosen for KV-quantization sensitivity, not for coverage), with upstream's
``synthetic.yaml`` arguments after ``num_needle_k = max(num_needle_k, num_needle_q)``:

  * ``niah_multikey_3`` - ``type_haystack=needle``, UUID keys AND UUID values, k=v=q=1. There is no
    lexical bridge between key and value, so the model cannot pattern-match its way to the answer.
    Because ``num_needle_q * num_needle_v == 1`` this is one of the variants whose template
    upstream rewrites into the singular (see :func:`singularise`).
  * ``niah_multiquery`` - ``type_haystack=essay``, word keys and 7-digit number values, k=4, v=1,
    q=4. Four reference strings give this item partial credit under ``string_match_all``.

Reproduced verbatim from upstream: the needle sentence, the task template AND its ``answer_prefix``
(see below), the singular rewrite, the fixed-seed shuffle of the needles (``Random(42)``), the
40-point ``DEPTHS`` ladder, the ``insertion_positions`` construction for the essay haystack, the
reverse-order index insertion for the ``needle``/``noise`` haystacks, the word pool
(``wonderwords`` adjective-noun compounds, all 6_171_620 of them - see ``vocabulary.py``), the
key/value types and the query/answer selection.

NAMED DIVERGENCES (weigh each before comparing a score with a published RULER number):

 1. **Answer prefix placement.** Upstream ``scripts/data/prepare.py`` appends ``answer_prefix`` to
    the template for *every* model type, and ``scripts/pred/call_api.py`` sends
    ``input + answer_prefix``; for a chat model the model's chat template is a literal string in
    ``scripts/data/template.py``, so the prefix lands *after* the assistant marker - upstream
    prefills the assistant turn. A chat-completions endpoint cannot prefill an assistant turn, so
    the identical prefix text is appended to the end of the single user turn instead. The bytes are
    upstream's; their position relative to the assistant marker is not.
 2. **Stable render.** Needle depths and haystack positions are drawn once per item and applied as
    fractions, so the item does not change while the length search runs; upstream re-draws content
    at every candidate size because its search advances the global RNG. For the essay haystack this
    is not an approximation at all - upstream's own placement is ``int(len(document_sents) * depth /
    100)`` from the same 40-point ``DEPTHS`` ladder, so one draw per item has exactly upstream's
    depth distribution. For the ``needle`` haystack upstream draws ``random.sample(range(
    num_haystack), len(needles))``, i.e. a uniform integer index; we draw a uniform fraction once
    and scale it, which is the same distribution up to rounding and is exact for the single-needle
    configuration ``niah_multikey_3`` actually uses.
 3. **Echo guards.** Keys and values are drawn distinctly and are rejected when they already occur
    in the haystack prose; upstream uses ``random.choice`` and checks nothing. Scoring is
    case-insensitive substring matching, so an unchecked draw can hand a model credit for echoing
    the context. The number of rejected draws is recorded per item in
    ``metadata["guard_rejections"]`` so the guard can never bias silently: for UUID and 7-digit
    values it is 0 in practice (a 7-digit literal in the essay corpus is rare, and the draw space is
    9*10^6 / 2^122).
 4. Sentence splitting and essay exhaustion - see ``corpus.py``.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .base import GeneratedItem, TokenCounter, count_messages, digest, user_messages
from .corpus import essay_words, sentence_split
from .fitting import DEFAULT_MAX_COUNTER_CALLS, fit_units
from .scoring import STRING_MATCH_ALL
from .vocabulary import niah_compound, niah_compound_count

TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize "
            "it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special "
            "magic {type_needle_v} for {query} mentioned in the provided text?")
"""``TASKS['niah']['template']`` in upstream ``scripts/data/synthetic/constants.py``."""
ANSWER_PREFIX = (" The special magic {type_needle_v} for {query} mentioned in the provided text are")
"""``TASKS['niah']['answer_prefix']``; ``prepare.py`` concatenates it onto the template."""
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
MAX_OUTPUT_TOKENS = 128
"""Upstream ``tokens_to_generate`` for the niah family."""
NEEDLE_SHUFFLE_SEED = 42
"""Upstream shuffles the needles with ``random.Random(args.random_seed)``, i.e. a fixed 42."""
DEPTHS: tuple[int, ...] = tuple(round(step * 100 / 39) for step in range(40))
"""``list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))`` - verified identical."""
_FRACTION_RESOLUTION = 1 << 20


@dataclass(frozen=True)
class NiahConfig:
    type_haystack: str
    type_needle_k: str
    type_needle_v: str
    num_needle_k: int
    num_needle_v: int
    num_needle_q: int


NIAH_CONFIGS: dict[str, NiahConfig] = {
    # ``synthetic.yaml`` declares niah_multiquery with num_needle_k=1; upstream's
    # ``args.num_needle_k = max(args.num_needle_k, args.num_needle_q)`` raises it to 4.
    "niah_multikey_3": NiahConfig("needle", "uuids", "uuids", 1, 1, 1),
    "niah_multiquery": NiahConfig("essay", "words", "numbers", 4, 1, 4),
}


def singularise(template: str, type_needle_v: str) -> tuple[str, str]:
    """Upstream's ``num_needle_q * num_needle_v == 1`` rewrite, applied to template+answer_prefix.

    ``'Some' -> 'A'``, ``'are all' -> 'is'``, ``'are' -> 'is'``, ``'answers' -> 'answer'``, and the
    trailing ``s`` is dropped from the needle-type word. Note that upstream applies this only to the
    *template*: the needle sentences keep the plural type word, because they are formatted from
    ``args.type_needle_v`` before the rewrite happens.
    """
    template = template.replace("Some", "A")
    template = template.replace("are all", "is")
    template = template.replace("are", "is")
    template = template.replace("answers", "answer")
    return template, type_needle_v[:-1]


def _draw(kind: str, rng: random.Random, taken: set[str], forbidden: str, rejections: list[int]) -> str:
    """Upstream's ``generate_random``, plus the two echo guards described in the module docstring."""
    for _ in range(64):
        if kind == "numbers":
            value = str(rng.randint(10 ** 6, 10 ** 7 - 1))
        elif kind == "uuids":
            value = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        elif kind == "words":
            value = niah_compound(rng.randrange(niah_compound_count()))
        else:
            raise ValueError(f"unknown RULER needle type: {kind!r}")
        if value not in taken and (not forbidden or value.lower() not in forbidden):
            taken.add(value)
            return value
        rejections[0] += 1
    raise ValueError(f"could not draw a distinct RULER {kind} value")


def _stream(make: Callable[[random.Random], str], seed: int) -> Callable[[int], list[str]]:
    """Prefix-stable deterministic stream: ``take(n)`` is always a prefix of ``take(n + k)``."""
    rng, cache = random.Random(seed), []

    def take(count: int) -> list[str]:
        while len(cache) < count:
            cache.append(make(rng))
        return cache[:count]

    return take


def _query_text(keys: list[str]) -> str:
    """``', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]``."""
    if len(keys) == 1:
        return keys[0]
    return ", ".join(keys[:-1]) + ", and " + keys[-1]


def _scaled_positions(fractions: list[int], units: int) -> list[int]:
    """Upstream's ``sorted(random.sample(range(num_haystack), len(needles)))``, from fixed draws."""
    ceiling = max(units - 1, 0)
    positions: list[int] = []
    for fraction in fractions:
        position = min(ceiling, fraction * units // _FRACTION_RESOLUTION)
        if positions and position <= positions[-1]:
            position = min(ceiling, positions[-1] + 1)
        positions.append(position)
    return positions


def generate(name: str, *, target_tokens: int, counter: TokenCounter, seed: int,
             dataset_root: str | None = None,
             max_counter_calls: int = DEFAULT_MAX_COUNTER_CALLS) -> GeneratedItem:
    config = NIAH_CONFIGS[name]
    rng = random.Random(seed)
    taken: set[str] = set()
    rejections = [0]
    # The haystack is resolved first so the needle draw can avoid anything that already occurs in
    # it; a value the prose already contains would be scored as "recovered" by an echo.
    forbidden, corpus_words = "", ()
    if config.type_haystack == "essay":
        if dataset_root is None:
            raise ValueError("the essay haystack needs the image's dataset root")
        corpus_words = essay_words(str(dataset_root))
        forbidden = " ".join(corpus_words).lower()
    elif config.type_haystack == "noise":
        forbidden = NOISE.lower()
    elif config.type_haystack != "needle":
        raise ValueError(f"unknown RULER haystack type: {config.type_haystack!r}")

    keys, values, needles = [], [], []
    for _ in range(config.num_needle_k):
        key = _draw(config.type_needle_k, rng, taken, forbidden, rejections)
        keys.append(key)
        per_key = []
        for _ in range(config.num_needle_v):
            value = _draw(config.type_needle_v, rng, taken, forbidden, rejections)
            per_key.append(value)
            needles.append(NEEDLE.format(type_needle_v=config.type_needle_v, key=key, value=value))
        values.append(per_key)
    random.Random(NEEDLE_SHUFFLE_SEED).shuffle(needles)

    if config.type_haystack == "essay":
        # Upstream: sorted([int(len(document_sents) * (depth / 100)) for depth in
        #                   random.sample(DEPTHS, len(needles))])
        depths = sorted(rng.sample(DEPTHS, len(needles)))
        depth_percent, max_units = list(depths), len(corpus_words)

        def render_context(units: int) -> str:
            sentences = sentence_split(" ".join(corpus_words[:units]))
            cuts = [0] + sorted(int(len(sentences) * (depth / 100)) for depth in depths) + [len(sentences)]
            pieces = []
            for index in range(1, len(cuts)):
                pieces.append(" ".join(sentences[cuts[index - 1]:cuts[index]]))
                if index - 1 < len(needles):
                    pieces.append(needles[index - 1])
            return " ".join(pieces)
    else:
        fractions = sorted(rng.sample(range(_FRACTION_RESOLUTION), len(needles)))
        depth_percent = [round(fraction * 100 / _FRACTION_RESOLUTION) for fraction in fractions]
        max_units = max(len(needles) + 1, target_tokens)
        if config.type_haystack == "noise":
            def lines(units: int) -> list[str]:
                return [NOISE] * units
        else:
            def make(stream_rng: random.Random) -> str:
                # ``taken`` is shared: a distractor can never repeat the real key or the real value.
                return NEEDLE.format(
                    type_needle_v=config.type_needle_v,
                    key=_draw(config.type_needle_k, stream_rng, taken, "", rejections),
                    value=_draw(config.type_needle_v, stream_rng, taken, "", rejections))

            take = _stream(make, rng.getrandbits(64))

            def lines(units: int) -> list[str]:
                return list(take(units))

        def render_context(units: int) -> str:
            # Upstream: indexes = sorted(random.sample(...), reverse=True); zip(indexes, needles).
            # The pairing really is descending-index against needle order; reproduced as-is.
            sentences = lines(units)
            for index, needle in zip(sorted(_scaled_positions(fractions, units), reverse=True), needles):
                sentences.insert(index, needle)
            return "\n".join(sentences)

    indices = rng.sample(range(config.num_needle_k), config.num_needle_q)
    query = _query_text([keys[index] for index in indices])
    answers = tuple(value for index in indices for value in values[index])

    template, type_needle_v = TEMPLATE + ANSWER_PREFIX, config.type_needle_v
    if config.num_needle_q * config.num_needle_v == 1:
        template, type_needle_v = singularise(template, type_needle_v)

    def render(units: int) -> tuple[dict[str, str], ...]:
        return user_messages(template.format(type_needle_v=type_needle_v, context=render_context(units),
                                             query=query))

    fit = fit_units(lambda units: count_messages(counter, render(units)), target_tokens=target_tokens,
                    max_units=max_units, max_counter_calls=max_counter_calls)
    messages = render(fit.units)
    metadata: dict[str, Any] = {
        "family": "retrieval", "type_haystack": config.type_haystack,
        "type_needle_k": config.type_needle_k, "type_needle_v": config.type_needle_v,
        "num_needle_k": config.num_needle_k, "num_needle_v": config.num_needle_v,
        "num_needle_q": config.num_needle_q, "insertion_depth_percent": depth_percent,
        "haystack_units": fit.units,
        "haystack_unit": "word" if config.type_haystack == "essay" else "line",
        "units_exhausted": fit.units >= max_units, "template_tokens": fit.base_tokens,
        "queried_keys": [keys[index] for index in indices], "guard_rejections": rejections[0],
        "singular_template": config.num_needle_q * config.num_needle_v == 1,
        "answer_prefix_in_prompt": True,
    }
    return GeneratedItem(name, seed, target_tokens, fit.tokens, fit.units, fit.counter_calls, messages,
                         answers, STRING_MATCH_ALL, MAX_OUTPUT_TOKENS, digest(list(messages)), metadata)
