"""RULER multi-hop variable tracking (``vt``).

Mirrors NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER) at commit
``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a``: ``scripts/data/synthetic/variable_tracking.py``, the
``variable_tracking`` entry of ``scripts/data/synthetic/constants.py`` and the ``vt`` entry of
``scripts/synthetic.yaml`` (``type_haystack=noise``, ``num_chains=1``, ``num_hops=4``).

One chain of ``num_hops`` assignments starts with a literal 5-digit value and then aliases it hop
by hop (``VAR ABCDE = 12345``, ``VAR FGHIJ = VAR ABCDE``...). The statements are scattered through
repeated noise sentences, in order, and the question asks which variables hold the queried value;
the reference answer is every variable in the chain, so ``string_match_all`` gives partial credit
per hop recovered. This is the most KV-sensitive task in the vendored set: a single damaged
key/value pair anywhere in the chain breaks every later hop.

Reproduced verbatim from upstream: the noise sentence, the assignment syntax (including the
trailing space upstream leaves after an aliasing statement), 5-uppercase-letter variable names,
values from ``randint(10000, 99999)``, ``num_chains=1``, ``num_hops=4``, the ``"\\n".join`` of the
noise lines followed by ``context.replace(". \\n", ".\\n")``, the ``sorted(random.sample(...))``
position draw with its ``+ j`` offset, the template, its ``answer_prefix`` and the ``num_v``
substitution (``num_hops + 1`` = 5).

NAMED DIVERGENCES:

 1. **Answer prefix placement** - identical to ``niah.py``'s divergence 1: upstream appends
    ``answer_prefix`` after the chat template's assistant marker (an assistant prefill); a
    chat-completions endpoint cannot prefill, so the same bytes end the single user turn. This
    matters more here than anywhere else, because upstream's ``tokens_to_generate`` for ``vt`` is
    **30** and the prefix ends with ``"they are: "``, which leaves the model nothing to do but name
    five variables. Without the prefix a chat model can spend the whole 30-token budget on a
    preamble; with it at the end of the user turn the instruction is present but the assistant turn
    still starts empty. ``output_cap_hit`` and ``finish_reason`` are on every row so a campaign can
    see whether the cap, rather than the KV cache, is what bit.
 2. **Stable render** - statement positions are drawn once as distinct fractions and rescaled at
    every candidate noise count, so the item does not change while the length search runs; upstream
    re-draws ``sorted(random.sample(range(num_noises), len(chain)))`` at each candidate size. Same
    distribution, same ``+ j`` offset, same ordering guarantee.
 3. **Echo guard** - variable names that already occur (case-insensitively) in the noise sentence or
    the template are rejected; upstream does not check. Scoring is case-insensitive substring
    matching, so ``GRASS`` or ``GREEN`` would be "recovered" by a model that merely echoes the
    prompt. The excluded set is every 5-character substring of the noise-plus-template text: 296 of
    the 26^5 = 11_881_376 possible names, i.e. 0.0025% of the draw space, and none of them is a
    name a model could confuse with another - the guard cannot bias the distribution measurably.
    The count is recorded per item in ``metadata["guard_excluded_names"]`` and pinned by a test.
 4. **Value RNG** - upstream draws the chain head from ``numpy.random.randint(10000, 99999)``
    (half-open) on a *separate* numpy stream from the ``random`` module stream it uses for names and
    positions. There is no numpy here; the same half-open range is drawn from this item's own
    ``random.Random``. Distribution identical, byte-for-byte reproduction of an upstream sample not
    possible (and not possible upstream either, across numpy versions).
 5. **Distinct names** - upstream draws ``(num_hops + 1) * num_chains`` names and then appends more
    until the *set* is large enough, while still slicing chains out of the original list, so a
    single chain can in principle repeat a name (probability ~8e-7 at 26^5). Here the names are
    distinct by construction.
"""

from __future__ import annotations

import random
import string
from typing import Any

from .base import GeneratedItem, TokenCounter, count_messages, digest, user_messages
from .fitting import DEFAULT_MAX_COUNTER_CALLS, fit_units
from .scoring import STRING_MATCH_ALL

TEMPLATE = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
            "{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.")
"""``TASKS['variable_tracking']['template']`` in upstream ``scripts/data/synthetic/constants.py``."""
ANSWER_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, {num_v} "
                 "variables are assigned the value {query}, they are: ")
"""``TASKS['variable_tracking']['answer_prefix']``; ``prepare.py`` concatenates it onto the template."""
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
MAX_OUTPUT_TOKENS = 30
"""Upstream ``tokens_to_generate`` for variable tracking."""
NUM_CHAINS = 1
NUM_HOPS = 4
VARIABLE_LENGTH = 5
_FRACTION_RESOLUTION = 1 << 20


def excluded_names() -> frozenset[str]:
    """Every 5-character window of the noise sentence and the template, upper-cased.

    These are the names the echo guard refuses; see NAMED DIVERGENCE 3. The set is a property of
    two fixed strings, so it is the same for every item and can be counted once.
    """
    text = (NOISE + " " + TEMPLATE + ANSWER_PREFIX).upper()
    return frozenset(text[start:start + VARIABLE_LENGTH]
                     for start in range(len(text) - VARIABLE_LENGTH + 1))


def generate_chains(num_chains: int, num_hops: int, rng: random.Random) -> tuple[list[list[str]], list[list[str]]]:
    """Upstream ``generate_chains``: distinct names, one literal head per chain, then aliases."""
    forbidden = excluded_names()
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < (num_hops + 1) * num_chains:
        name = "".join(rng.choices(string.ascii_uppercase, k=VARIABLE_LENGTH))
        if name not in seen and name not in forbidden:
            seen.add(name)
            names.append(name)
    variables, chains = [], []
    for start in range(0, len(names), num_hops + 1):
        chain_names = names[start:start + num_hops + 1]
        value = rng.randrange(10000, 99999)  # numpy's randint is half-open; see DIVERGENCE 4
        statements = [f"VAR {chain_names[0]} = {value}"]
        for index in range(1, len(chain_names)):
            statements.append(f"VAR {chain_names[index]} = VAR {chain_names[index - 1]} ")
        variables.append(chain_names)
        chains.append(statements)
    return variables, chains


def generate(name: str, *, target_tokens: int, counter: TokenCounter, seed: int,
             dataset_root: str | None = None, max_counter_calls: int = DEFAULT_MAX_COUNTER_CALLS,
             num_chains: int = NUM_CHAINS, num_hops: int = NUM_HOPS) -> GeneratedItem:
    if name != "vt":
        raise ValueError(f"unknown variable-tracking task: {name!r}")
    rng = random.Random(seed)
    variables, chains = generate_chains(num_chains, num_hops, rng)
    fractions = [sorted(rng.sample(range(_FRACTION_RESOLUTION), len(chain))) for chain in chains]
    query = chains[0][0].split("=")[-1].strip()
    answers = tuple(variables[0])
    template = (TEMPLATE + ANSWER_PREFIX).replace("{num_v}", str(num_hops + 1))
    max_units = max(len(chains[0]) + 1, target_tokens)

    def render(units: int) -> tuple[dict[str, str], ...]:
        sentences = [NOISE] * units
        ceiling = max(units - 1, 0)
        for chain, chain_fractions in zip(chains, fractions):
            for offset, (fraction, statement) in enumerate(zip(chain_fractions, chain)):
                sentences.insert(min(ceiling, fraction * units // _FRACTION_RESOLUTION) + offset, statement)
        context = "\n".join(sentences).replace(". \n", ".\n")
        return user_messages(template.format(context=context, query=query))

    fit = fit_units(lambda units: count_messages(counter, render(units)), target_tokens=target_tokens,
                    max_units=max_units, max_counter_calls=max_counter_calls)
    messages = render(fit.units)
    metadata: dict[str, Any] = {
        "family": "multi_hop", "num_chains": num_chains, "num_hops": num_hops, "query_value": query,
        "noise_sentences": fit.units, "haystack_unit": "noise_sentence",
        "units_exhausted": fit.units >= max_units, "template_tokens": fit.base_tokens,
        "decoy_values": [chain[0].split("=")[-1].strip() for chain in chains[1:]],
        "guard_excluded_names": len(excluded_names()), "answer_prefix_in_prompt": True,
    }
    return GeneratedItem("vt", seed, target_tokens, fit.tokens, fit.units, fit.counter_calls, messages,
                         answers, STRING_MATCH_ALL, MAX_OUTPUT_TOKENS, digest(list(messages)), metadata)
