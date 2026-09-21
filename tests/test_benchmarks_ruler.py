"""Offline tests for the vendored RULER generators, scorer and adapter.

No Docker, no model, no GPU, no network: a fake whitespace tokenizer stands in for the counting
route and a fake transport stands in for the endpoint. Generated lengths stay in the hundreds of
tokens so the whole file runs in seconds while exercising the same code paths a 128K item would.

The ``upstream fidelity`` section pins the vendored strings and parameters against text fetched
from NVIDIA RULER at commit ``c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`` and stored under
``tests/data/ruler-upstream-*``. Those fixtures are the reference, not a paraphrase of it: the
scorer is checked by executing upstream's own three functions side by side with ours.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.benchmarks import BenchmarkAborted, BenchmarkContext, REQUIRED_SAMPLE_KEYS
from llmbench.benchmarks.ruler import (
    DEFAULT_LENGTHS, DEFAULT_TASKS, RulerAdapter, effective_length, estimated_item_seconds,
    length_curve, parse_task_id, task_id_for,
)
from llmbench.benchmarks.ruler_tasks import (
    DEFERRED_TASKS, TASK_SPECS, CorpusUnavailable, LengthUnreachable, digest, generate_task,
    postprocess_pred, score_prediction, string_match_all, string_match_part,
)
from llmbench.benchmarks.ruler_tasks import common_words, niah, variable_tracking, wonderwords_lists
from llmbench.benchmarks.ruler_tasks.common_words import FREQ_CW, FREQ_UCW, LONG_BRANCH, NUM_CW
from llmbench.benchmarks.ruler_tasks.corpus import ESSAY_JSON, essay_sentences, essay_words
from llmbench.benchmarks.ruler_tasks.vocabulary import (
    cwe_pool_size, cwe_word_pool, niah_compound, niah_compound_count,
)

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "ruler-essays.json"
CORPUS_FREE = ("niah_multikey_3", "vt", "cwe")
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
UPSTREAM_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
UPSTREAM = json.loads((DATA / "ruler-upstream-templates.json").read_text(encoding="utf-8"))
UPSTREAM_WORDS = json.loads((DATA / "ruler-upstream-wordlists.json").read_text(encoding="utf-8"))
CONTEXT_SENTINEL = "\x00CONTEXT\x00"


def upstream_frame(task: str, item) -> tuple[str, str]:
    """Rebuild upstream's prompt for this item from the fetched fixture, around ``{context}``.

    ``prepare.py`` concatenates ``template + answer_prefix`` for every model type and
    ``call_api.py`` sends ``input + answer_prefix``, so the full upstream prompt for one sample is
    the concatenation - this returns the part before and the part after the context it wraps.
    """
    if task.startswith("niah_"):
        entry = UPSTREAM["tasks"]["niah"]
        template = entry["template"] + entry["answer_prefix"]
        type_needle_v = item.metadata["type_needle_v"]
        if item.metadata["num_needle_q"] * item.metadata["num_needle_v"] == 1:
            for old, new in (("Some", "A"), ("are all", "is"), ("are", "is"), ("answers", "answer")):
                template = template.replace(old, new)
            type_needle_v = type_needle_v[:-1]
        keys = item.metadata["queried_keys"]
        query = keys[0] if len(keys) == 1 else ", ".join(keys[:-1]) + ", and " + keys[-1]
        rendered = template.format(type_needle_v=type_needle_v, context=CONTEXT_SENTINEL, query=query)
    elif task == "vt":
        entry = UPSTREAM["tasks"]["variable_tracking"]
        rendered = (entry["template"] + entry["answer_prefix"]).format(
            context=CONTEXT_SENTINEL, query=item.metadata["query_value"],
            num_v=item.metadata["num_hops"] + 1)
    else:
        entry = UPSTREAM["tasks"]["common_words_extraction"]
        rendered = (entry["template"] + entry["answer_prefix"]).format(context=CONTEXT_SENTINEL)
    head, _, tail = rendered.partition(CONTEXT_SENTINEL)
    return head, tail


class FakeCounter:
    """One fixture token per whitespace word, plus a per-message template allowance.

    Deliberately not a real tokenizer: it only has to be deterministic and monotonic in the prompt
    so the length search can be tested without loading a model.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.saw_tools: list = []

    def __call__(self, messages, tools):
        self.calls += 1
        self.saw_tools.append(tools)
        assert all(set(message) == {"role", "content"} for message in messages)
        return {"tokens": sum(len(message["content"].split()) + 4 for message in messages) + 1,
                "method": "fixture"}


class RecordingArtifacts:
    def __init__(self, fail_on: str | None = None) -> None:
        self.writes: dict[str, bytes] = {}
        self.fail_on = fail_on

    def write(self, relative: str, content: bytes) -> None:
        if self.fail_on is not None and self.fail_on in relative:
            raise OSError("no space left on device")
        if relative in self.writes:
            raise FileExistsError(relative)
        assert type(content) is bytes
        self.writes[relative] = content


class OracleTransport:
    """Answers each generated prompt from the item it belongs to; no network, no model."""

    def __init__(self, items, *, token_delta: int = 0, truncation: bool | None = False,
                 policy: str = "reject-overflow-no-shift", fail_on: tuple[int, ...] = (),
                 answer=None, served_model: str = "candidate") -> None:
        self.by_prompt = {item.prompt_sha256: item for item in items.values()}
        self.token_delta, self.truncation, self.policy = token_delta, truncation, policy
        self.fail_on, self.served_model = fail_on, served_model
        self.answer = answer or (lambda item: "The answer is " + ", ".join(item.answers) + ".")
        self.calls: list[dict] = []

    def __call__(self, payload: dict):
        self.calls.append(payload)
        if len(self.calls) in self.fail_on:
            raise RuntimeError("the endpoint dropped the connection")
        item = self.by_prompt[digest(payload["messages"])]
        response = {"model": self.served_model, "usage": {
            "prompt_tokens": item.actual_tokens + self.token_delta, "completion_tokens": 9},
            "choices": [{"message": {"role": "assistant", "content": self.answer(item)},
                         "finish_reason": "stop"}],
            "llmbench_context_policy": self.policy}
        if self.truncation is not None:
            response["input_truncated"] = self.truncation
        return response


@pytest.fixture
def dataset_root(tmp_path) -> str:
    root = tmp_path / "datasets"
    (root / "ruler").mkdir(parents=True)
    shutil.copyfile(FIXTURE, root / ESSAY_JSON)
    return str(root)


def make_context(*, dataset_root: str | None = None, task_ids=(), split="development", seed=11,
                 options=None, remaining=None, artifacts=None) -> BenchmarkContext:
    return BenchmarkContext(
        base_url="http://inference:8080/v1", model_alias="candidate", task_ids=tuple(task_ids),
        split=split, seed=seed,
        generation=SimpleNamespace(temperature=0.0, top_p=1.0, seed=7, max_output_tokens=256, top_k=None),
        remaining_seconds=remaining or (lambda: 10_000.0), artifacts=artifacts or RecordingArtifacts(),
        session_lock=SimpleNamespace(), dataset_root=dataset_root or "", options=dict(options or {}))


def wired_options(counter, transport, **extra):
    return {"token_counter": counter, "completion": transport, "context_capacity": 262_144,
            "tokenizer_verified": True, "tokenizer_id": "fixture-tokenizer",
            "template_id": "fixture-template", **extra}


def build_items(adapter: RulerAdapter, context: BenchmarkContext, counter) -> dict:
    """Regenerate every declared item so the fake transport can answer it (also proves determinism)."""
    items = {}
    for task_id in adapter.task_ids(context):
        task, length = parse_task_id(task_id)
        items[task_id] = generate_task(task, target_tokens=length, counter=counter,
                                       seed=adapter.item_seed(context, task, length),
                                       dataset_root=context.dataset_root or None)
    return items


# ---- upstream fidelity -------------------------------------------------------------------------

def test_vendored_template_strings_are_byte_identical_to_upstream():
    """Every prompt string and cap, against text fetched from RULER at the pinned commit."""
    assert UPSTREAM["_commit"] == UPSTREAM_WORDS["_commit"] == UPSTREAM_COMMIT
    niah_task, vt_task = UPSTREAM["tasks"]["niah"], UPSTREAM["tasks"]["variable_tracking"]
    cwe_task = UPSTREAM["tasks"]["common_words_extraction"]
    assert niah.TEMPLATE == niah_task["template"]
    assert niah.ANSWER_PREFIX == niah_task["answer_prefix"]
    assert niah.NEEDLE == UPSTREAM["niah_needle"]
    assert niah.NOISE == variable_tracking.NOISE == UPSTREAM["noise_sentence"]
    assert variable_tracking.TEMPLATE == vt_task["template"]
    assert variable_tracking.ANSWER_PREFIX == vt_task["answer_prefix"]
    assert common_words.TEMPLATE == cwe_task["template"]
    assert common_words.ANSWER_PREFIX == cwe_task["answer_prefix"]
    assert list(niah.DEPTHS) == UPSTREAM["depths"] and len(set(niah.DEPTHS)) == 40
    for task, family in (("niah_multikey_3", "niah"), ("niah_multiquery", "niah"),
                         ("vt", "variable_tracking"), ("cwe", "common_words_extraction")):
        assert TASK_SPECS[task].max_output_tokens == UPSTREAM["tasks"][family]["tokens_to_generate"]
        assert TASK_SPECS[task].metric == UPSTREAM["metrics"][family]
    assert (common_words.FREQ_CW, common_words.FREQ_UCW, common_words.NUM_CW) == (30, 3, 10)


def test_vendored_task_parameters_match_upstream_synthetic_yaml():
    for task, config in niah.NIAH_CONFIGS.items():
        args = UPSTREAM["task_args"][task]
        assert args["task"] == "niah"
        assert config.type_haystack == args["type_haystack"]
        assert (config.type_needle_k, config.type_needle_v) == (args["type_needle_k"], args["type_needle_v"])
        # niah.py: args.num_needle_k = max(args.num_needle_k, args.num_needle_q)
        assert config.num_needle_k == max(args["num_needle_k"], args["num_needle_q"])
        assert (config.num_needle_v, config.num_needle_q) == (args["num_needle_v"], args["num_needle_q"])
    vt_args = UPSTREAM["task_args"]["vt"]
    assert (variable_tracking.NUM_CHAINS, variable_tracking.NUM_HOPS) == (vt_args["num_chains"],
                                                                          vt_args["num_hops"])
    assert vt_args["type_haystack"] == "noise" and variable_tracking.VARIABLE_LENGTH == 5
    cwe_args = UPSTREAM["task_args"]["cwe"]
    assert (common_words.FREQ_CW, common_words.FREQ_UCW, common_words.NUM_CW) == (
        cwe_args["freq_cw"], cwe_args["freq_ucw"], cwe_args["num_cw"])
    # Every task upstream declares is either vendored or named as deferred, never silently dropped.
    assert set(UPSTREAM["task_args"]) == set(TASK_SPECS) | set(DEFERRED_TASKS)
    assert not set(TASK_SPECS) & set(DEFERRED_TASKS)


@pytest.mark.parametrize("task", sorted(TASK_SPECS))
def test_every_generated_prompt_is_upstreams_template_around_its_context(task, dataset_root):
    """The rendered prompt must equal upstream's template+answer_prefix wrapped round a context."""
    item = generate_task(task, target_tokens=1024, counter=FakeCounter(), seed=21,
                         dataset_root=dataset_root)
    head, tail = upstream_frame(task, item)
    assert len(head) > 40 and len(tail) > 20
    assert item.prompt.startswith(head), (task, item.prompt[:len(head) + 20])
    assert item.prompt.endswith(tail), (task, item.prompt[-len(tail) - 20:])
    assert item.metadata["answer_prefix_in_prompt"] is True
    # The prefix is upstream's bytes in a position a chat endpoint can reach: end of the user turn.
    assert [message["role"] for message in item.messages] == ["user"]


def test_vendored_scorer_agrees_with_upstreams_own_implementation():
    reference: dict = {}
    source = (DATA / "ruler-upstream-scoring.py").read_text(encoding="utf-8")
    assert UPSTREAM_COMMIT in source
    # Executing a pinned, in-repo fixture: the point is to score against upstream's own code.
    exec(compile(source, "ruler-upstream-scoring.py", "exec"), reference)
    cases = [
        ("bravo and delta", ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
        ("ALPHA bravo Charlie delta", ["alpha", "BRAVO", "charlie", "Delta"]),
        ("nothing at all", ["ALPHA", "BRAVO"]),
        ("  padded\x00answer  ", ["padded", "answer"]),
        ("1. ai 2. be", ["ai", "be", "ci"]),
        ("", ["ALPHA"]),
    ]
    for prediction, references in cases:
        ours = postprocess_pred(prediction)
        assert ours == reference["postprocess_pred"](prediction, {})
        assert round(string_match_all(ours, references) * 100, 2) == reference["string_match_all"](
            [ours], [references])
        assert round(string_match_part(ours, references) * 100, 2) == reference["string_match_part"](
            [ours], [references])


def test_vendored_word_pools_are_upstreams_pools():
    digests = wonderwords_lists.verify()
    for name, info in UPSTREAM_WORDS["wonderwords"].items():
        assert digests[name] == info["sha256"]
    assert len(wonderwords_lists.adjectives()) == UPSTREAM_WORDS["wonderwords"]["adjectivelist"]["entries"]
    assert len(wonderwords_lists.nouns()) == UPSTREAM_WORDS["wonderwords"]["nounlist"]["entries"]
    assert len(wonderwords_lists.verbs()) == UPSTREAM_WORDS["wonderwords"]["verblist"]["entries"]
    # cwe draws single words; only niah joins them into adjective-noun compounds.
    assert cwe_pool_size() == UPSTREAM_WORDS["pools"]["cwe"]["size"] == 8166
    assert niah_compound_count() == UPSTREAM_WORDS["pools"]["niah_words"]["size"] == 6_171_620
    assert set(cwe_word_pool()) == set(wonderwords_lists.nouns()) | set(
        wonderwords_lists.adjectives()) | set(wonderwords_lists.verbs())
    assert len(set(cwe_word_pool())) == len(cwe_word_pool()), "the pool is deduplicated"
    # sorted(...) then Random(42).shuffle: not sorted any more, but a fixed permutation of a sort.
    assert list(cwe_word_pool()) != sorted(cwe_word_pool())
    assert cwe_word_pool() == cwe_word_pool()


def test_compound_indexing_reproduces_upstreams_sorted_compound_list():
    """``niah_compound`` must be ``sorted(set(f'{adj}-{noun}'))[i]`` without building 6.2M strings."""
    adjectives = sorted(set(wonderwords_lists.adjectives()))
    nouns = sorted(set(wonderwords_lists.nouns()))
    brute = sorted({f"{adjective}-{noun}" for adjective in adjectives[:60] for noun in nouns[:60]})
    assert brute == [f"{adjective}-{noun}" for adjective in adjectives[:60] for noun in nouns[:60]]
    assert niah_compound(0) == f"{adjectives[0]}-{nouns[0]}"
    assert niah_compound(niah_compound_count() - 1) == f"{adjectives[-1]}-{nouns[-1]}"
    assert niah_compound(len(nouns)) == f"{adjectives[1]}-{nouns[0]}"
    probes = [0, 1, len(nouns) - 1, len(nouns), 5_000_000, niah_compound_count() - 1]
    assert [niah_compound(index) for index in probes] == sorted(niah_compound(i) for i in probes)
    for bad in (-1, niah_compound_count(), 1.5):
        with pytest.raises(ValueError):
            niah_compound(bad)


def test_essay_haystack_unit_is_upstreams_word_not_our_sentence(dataset_root):
    """Upstream slices the essay into words; sentences only decide where a needle is inserted."""
    words = essay_words(dataset_root)
    assert len(words) > len(essay_sentences(dataset_root)) > 64
    assert all(" " not in word for word in words)
    item = generate_task("niah_multiquery", target_tokens=1024, counter=FakeCounter(), seed=7,
                         dataset_root=dataset_root)
    assert item.metadata["haystack_unit"] == "word"
    assert item.metadata["haystack_units"] == item.units
    assert all(depth in UPSTREAM["depths"] for depth in item.metadata["insertion_depth_percent"])


def test_echo_guards_are_recorded_and_bounded():
    """Guards upstream does not have must be visible and provably tiny, never silent."""
    item = generate_task("niah_multikey_3", target_tokens=1024, counter=FakeCounter(), seed=12)
    assert item.metadata["guard_rejections"] == 0, "a UUID draw should never collide in practice"
    excluded = variable_tracking.excluded_names()
    assert len(excluded) == 296 and all(len(name) == 5 for name in excluded)
    # These are exactly the 5-character windows of the noise sentence and the template.
    assert {"GRASS", "GREEN", "THERE", "ABOVE"} <= excluded and "QTUYT" not in excluded
    assert len(excluded) / 26 ** 5 < 1e-4  # 0.0025% of the draw space
    vt_item = generate_task("vt", target_tokens=1024, counter=FakeCounter(), seed=12)
    assert vt_item.metadata["guard_excluded_names"] == len(excluded)
    assert not set(vt_item.answers) & excluded


# ---- generators --------------------------------------------------------------------------------

@pytest.mark.parametrize("task", sorted(TASK_SPECS))
def test_every_generator_hits_its_target_and_scores_itself(task, dataset_root):
    counter, target = FakeCounter(), 1024
    item = generate_task(task, target_tokens=target, counter=counter, seed=5, dataset_root=dataset_root)
    assert item.task == task and item.answers
    assert item.actual_tokens <= target, "the search must never accept a prompt above the target"
    assert target - item.actual_tokens <= 64, (task, item.actual_tokens)
    assert item.actual_tokens == counter([dict(m) for m in item.messages], [])["tokens"]
    assert counter.saw_tools and all(tools == [] for tools in counter.saw_tools)
    assert item.counter_calls <= 24 and item.metadata["template_tokens"] <= item.actual_tokens
    perfect = score_prediction(", ".join(item.answers), item.answers, metric=item.metric)
    assert perfect.score == 1.0 and perfect.correct
    assert score_prediction("nothing relevant here", item.answers, metric=item.metric).score == 0.0
    # The persisted record is hashes plus bounded previews: a 128K prompt never lands in an artifact.
    summary = item.summary()
    assert summary["prompt_sha256"] == item.prompt_sha256 and "prompt" not in summary
    assert len(summary["prompt_head"]) <= 240 and len(summary["prompt_tail"]) <= 240
    assert len(json.dumps(summary)) < len(item.prompt)


def test_generated_prompt_is_a_single_user_turn_without_a_leaked_answer_key(dataset_root):
    item = generate_task("niah_multiquery", target_tokens=1024, counter=FakeCounter(), seed=3,
                         dataset_root=dataset_root)
    assert [message["role"] for message in item.messages] == ["user"]
    for answer in item.answers:
        assert item.prompt.count(answer) == 1, "each value must be stored exactly once in the haystack"


def test_scorer_matches_upstream_substring_semantics():
    references = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    assert string_match_all("bravo and delta", references) == 0.5
    assert string_match_all("alpha bravo charlie delta", references) == 1.0
    assert string_match_all("nothing", references) == 0.0
    assert string_match_part("delta", references) == 1.0
    assert string_match_part("nothing", references) == 0.0
    partial = score_prediction("Only bravo", references)
    assert partial.score == 0.25 and partial.matched == ("BRAVO",) and not partial.correct
    assert [row["present"] for row in partial.per_expected] == [False, True, False, False]
    assert score_prediction("Only bravo", references, metric="string_match_part").score == 1.0
    assert postprocess_pred("  a\x00b  ") == "a\nb" and postprocess_pred(None) == ""
    assert score_prediction("", references).prediction_empty
    with pytest.raises(ValueError):
        score_prediction("x", references, metric="llm_judge")
    with pytest.raises(ValueError):
        score_prediction("x", [])


def test_lenient_rendering_reports_beside_the_strict_score_and_never_replaces_it():
    references = ["ALPHA"]
    fenced = "```json\n[\"ALPHA\"]\n```"
    assert score_prediction(fenced, references).score == 1.0  # a fence does not hide a substring
    spaced = "AL\nPHA"
    assert score_prediction(spaced, references).score == 0.0
    assert score_prediction(spaced, references, lenient=True).score == 0.0  # no semantic repair


def test_multikey_3_uuid_needles_have_no_lexical_bridge_between_key_and_value():
    item = generate_task("niah_multikey_3", target_tokens=1024, counter=FakeCounter(), seed=9)
    key = item.metadata["queried_keys"][0]
    value = item.answers[0]
    assert UUID_PATTERN.match(key) and UUID_PATTERN.match(value) and key != value
    grams = {key[index:index + 6] for index in range(len(key) - 5) if "-" not in key[index:index + 6]}
    assert not any(gram in value for gram in grams), "a shared substring would be a lexical shortcut"
    # Stored once; the key is named once in the needle, once in the question and once in upstream's
    # answer prefix (``prepare.py`` appends the prefix for every model type - see base.user_messages).
    assert item.prompt.count(value) == 1 and item.prompt.count(key) == 3
    haystack_values = re.findall(r"is: ([0-9a-f-]{36})\.", item.prompt)
    assert len(haystack_values) == len(set(haystack_values)) > 10
    assert item.metadata["type_haystack"] == "needle"
    # num_needle_q * num_needle_v == 1, so upstream rewrites the template into the singular.
    assert item.metadata["singular_template"] and "A special magic uuid is hidden" in item.prompt
    assert "What is the special magic uuid for" in item.prompt and "uuids" not in item.prompt.split("\n")[0]


def test_vt_chain_resolves_hop_by_hop_to_the_queried_value():
    item = generate_task("vt", target_tokens=1024, counter=FakeCounter(), seed=4)
    chain = list(item.answers)
    assert len(chain) == len(set(chain)) == item.metadata["num_hops"] + 1
    assert all(len(name) == 5 and name.isupper() for name in chain)
    query = item.metadata["query_value"]
    assert item.prompt.count(f"VAR {chain[0]} = {query}") == 1
    for previous, current in zip(chain, chain[1:]):
        assert f"VAR {current} = VAR {previous}" in item.prompt
    # Three times: the chain head's assignment, the question, and upstream's answer prefix.
    assert item.prompt.count(query) == 3 and f"assigned the value {query}" in item.prompt
    assert item.prompt.count(f"= {query}") == 1, "no second variable may hold the queried value"
    # Upstream joins the noise lines with "\n" and substitutes num_v = num_hops + 1 into the prefix.
    assert item.prompt.count("\n" + variable_tracking.NOISE) >= 10
    assert item.prompt.endswith("5 variables are assigned the value "
                                f"{query}, they are: ")
    # Every hop is a separate reference string, so a chain broken halfway scores as partial credit.
    assert score_prediction(", ".join(chain[:3]), item.answers).score == pytest.approx(3 / 5)


def test_cwe_answer_words_are_exactly_the_over_represented_ones():
    item = generate_task("cwe", target_tokens=8192, counter=FakeCounter(), seed=6)
    assert not item.metadata["short_context_branch"]
    # Upstream prepends num_fewshot=1 worked example and joins it to the question with "\n";
    # only the final list is the one the question is about, so count only that one.
    shot, _, question = item.prompt.partition("\n" + common_words.TEMPLATE.split("{context}")[0])
    assert question and item.metadata["num_fewshot"] == 1
    assert shot.startswith(common_words.TEMPLATE.split("{context}")[0])
    words = re.findall(r"\d+\. ([a-zA-Z-]+)", question)
    counts = {word: words.count(word) for word in set(words)}
    assert len(item.answers) == NUM_CW
    assert all(counts[answer] == FREQ_CW for answer in item.answers)
    assert {count for word, count in counts.items() if word not in item.answers} == {FREQ_UCW}
    assert item.metadata["total_words"] == len(words)
    assert item.metadata["num_words"] == NUM_CW + item.metadata["uncommon_words"]
    # The shot demonstrates the answer format upstream's substring metric is looking for.
    shot_answer = shot.rsplit(common_words.ANSWER_PREFIX, 1)[-1]
    assert len(re.findall(r"\d+\. ([a-zA-Z-]+)", shot_answer)) == NUM_CW
    assert set(re.findall(r"\d+\. ([a-zA-Z-]+)", shot)) <= set(cwe_word_pool())


def test_cwe_short_context_branch_uses_upstreams_easier_repeat_counts():
    item = generate_task("cwe", target_tokens=2048, counter=FakeCounter(), seed=6)
    assert item.metadata["short_context_branch"]
    assert (item.metadata["freq_cw"], item.metadata["freq_ucw"]) == (6, 1)
    assert item.metadata["fewshot_words"] == 20
    long_item = generate_task("cwe", target_tokens=8192, counter=FakeCounter(), seed=6)
    assert (long_item.metadata["freq_cw"], long_item.metadata["freq_ucw"]) == (FREQ_CW, FREQ_UCW)
    assert long_item.metadata["fewshot_words"] == LONG_BRANCH["shot_words"] == 40


def test_seeds_are_reproducible_and_different_seeds_give_different_items(dataset_root):
    for task in sorted(TASK_SPECS):
        first = generate_task(task, target_tokens=1024, counter=FakeCounter(), seed=17,
                              dataset_root=dataset_root)
        again = generate_task(task, target_tokens=1024, counter=FakeCounter(), seed=17,
                              dataset_root=dataset_root)
        other = generate_task(task, target_tokens=1024, counter=FakeCounter(), seed=18,
                              dataset_root=dataset_root)
        assert first.prompt_sha256 == again.prompt_sha256 and first.answers == again.answers
        assert first.prompt_sha256 != other.prompt_sha256 and first.answers != other.answers


def test_development_and_holdout_items_are_disjoint(dataset_root):
    adapter = RulerAdapter(tasks=CORPUS_FREE, lengths=(1024,))
    development = make_context(dataset_root=dataset_root, split="development")
    holdout = make_context(dataset_root=dataset_root, split="holdout")
    assert adapter.task_ids(development) == adapter.task_ids(holdout)  # same declared matrix
    for task_id in adapter.task_ids(development):
        task, length = parse_task_id(task_id)
        seeds = {adapter.item_seed(context, task, length) for context in (development, holdout)}
        assert len(seeds) == 2
        items = [generate_task(task, target_tokens=length, counter=FakeCounter(), seed=seed)
                 for seed in sorted(seeds)]
        assert items[0].prompt_sha256 != items[1].prompt_sha256
        assert not set(items[0].answers) & set(items[1].answers)


def test_length_search_refuses_an_impossible_target_and_bounds_its_counting():
    with pytest.raises(LengthUnreachable):
        generate_task("cwe", target_tokens=16, counter=FakeCounter(), seed=1)
    counter = FakeCounter()
    generate_task("vt", target_tokens=2048, counter=counter, seed=1, max_counter_calls=6)
    assert counter.calls <= 6


def test_a_haystack_too_small_for_the_requested_length_is_reported_not_padded(dataset_root):
    """The baked essay dump must be large enough; a short one fails closed instead of repeating."""
    item = generate_task("niah_multiquery", target_tokens=20_000, counter=FakeCounter(), seed=2,
                         dataset_root=dataset_root)
    assert item.metadata["units_exhausted"] and item.actual_tokens < 20_000
    adapter = RulerAdapter(tasks=("niah_multiquery",), lengths=(20_000,))
    transport = OracleTransport({})
    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(FakeCounter(), transport)))
    assert transport.calls == [] and rows[0]["outcome_status"] == "length_target_unmet"
    assert rows[0]["ruler_metadata"]["units_exhausted"] and rows[0]["status"] == "environment_error"


def test_essay_haystack_absence_is_reported_and_never_substituted(tmp_path, dataset_root):
    with pytest.raises(CorpusUnavailable, match="PaulGrahamEssays.json"):
        generate_task("niah_multiquery", target_tokens=1024, counter=FakeCounter(), seed=1,
                      dataset_root=str(tmp_path / "empty"))
    assert len(essay_sentences(dataset_root)) > 64


# ---- adapter -----------------------------------------------------------------------------------

def test_task_ids_encode_task_and_length_and_round_trip(dataset_root):
    adapter = RulerAdapter()
    ids = adapter.task_ids(make_context(dataset_root=dataset_root))
    assert len(ids) == len(DEFAULT_TASKS) * len(DEFAULT_LENGTHS) == len(set(ids))
    assert task_id_for("vt", 131072) == "ruler/vt/131072" and "ruler/vt/131072" in ids
    assert [parse_task_id(task_id)[1] for task_id in ids] == sorted(parse_task_id(i)[1] for i in ids)
    for bad in ("ruler/vt", "niah/vt/4096", "ruler/vt/0", "ruler/vt/many"):
        with pytest.raises(ValueError):
            parse_task_id(bad)
    declared = make_context(dataset_root=dataset_root, task_ids=("ruler/cwe/2048", "ruler/vt/1024"))
    assert adapter.task_ids(declared) == ("ruler/vt/1024", "ruler/cwe/2048")


def test_available_is_false_without_the_corpus_and_true_for_corpus_free_tasks(tmp_path, dataset_root):
    adapter = RulerAdapter(lengths=(1024,))
    empty = make_context(dataset_root=str(tmp_path / "nothing"))
    ok, reason = adapter.available(empty)
    assert not ok and "PaulGrahamEssays.json" in reason and "missing" in reason
    ok, reason = adapter.available(make_context(dataset_root=dataset_root))
    assert ok and "ruler-vendored-v1" in reason
    corpus_free = RulerAdapter(tasks=CORPUS_FREE, lengths=(1024,))
    ok, reason = corpus_free.available(empty)
    assert ok, reason
    ok, reason = adapter.available(make_context(dataset_root=dataset_root, task_ids=("ruler/qa_2/4096",)))
    assert not ok and DEFERRED_TASKS["qa_2"] in reason
    ok, reason = adapter.available(make_context(dataset_root=dataset_root, task_ids=("nonsense",)))
    assert not ok and "preflight failed" in reason


def test_a_run_produces_one_scored_row_per_declared_task(dataset_root):
    adapter = RulerAdapter(tasks=("vt", "cwe"), lengths=(1024, 2048))
    counter, artifacts = FakeCounter(), RecordingArtifacts()
    context = make_context(dataset_root=dataset_root, artifacts=artifacts)
    items = build_items(adapter, context, FakeCounter())
    transport = OracleTransport(items)
    context = make_context(dataset_root=dataset_root, artifacts=artifacts,
                           options=wired_options(counter, transport))
    rows = adapter.run(context)
    assert [row["task_id"] for row in rows] == list(adapter.task_ids(context))
    for row in rows:
        assert all(key in row for key in REQUIRED_SAMPLE_KEYS)
        assert row["status"] == "completed" and row["outcome_status"] == "passed"
        assert row["score"] == 1.0 and row["passed"] and row["model_evaluated"] and not row["synthetic"]
        assert row["suite"] == "ruler" and row["category"] == "retrieval"
        assert row["target_input_tokens"] == parse_task_id(row["task_id"])[1]
        assert row["length_within_tolerance"] and row["actual_input_tokens"] <= row["target_input_tokens"]
        assert row["eligible_for_context_claim"] and row["context"]["verification_status"] == "verified"
        assert row["prompt_sha256"] == items[row["task_id"]].prompt_sha256
        slug = row["task_id"].replace("/", "-")
        assert f"ruler/{slug}/prompt.json" in artifacts.writes
        assert f"ruler/{slug}/response.json" in artifacts.writes
    prompt_record = json.loads(artifacts.writes["ruler/ruler-vt-1024/prompt.json"])
    assert prompt_record["prompt_sha256"] and prompt_record["prompt_characters"] > 1000
    assert "prompt" not in prompt_record and len(prompt_record["prompt_head"]) <= 240
    assert [payload["max_tokens"] for payload in transport.calls] == [
        TASK_SPECS[parse_task_id(row["task_id"])[0]].max_output_tokens for row in rows]
    assert all(payload["temperature"] == 0.0 and payload["seed"] == 7 for payload in transport.calls)


def test_partial_credit_and_degradation_are_visible_per_length(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024, 2048))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())

    def decaying(item):  # the long item recovers only the first two hops
        answers = item.answers if item.target_tokens == 1024 else item.answers[:2]
        return ", ".join(answers)

    transport = OracleTransport(items, answer=decaying)
    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(FakeCounter(), transport)))
    short, long = rows
    assert short["score"] == 1.0 and short["outcome_status"] == "passed"
    assert long["score"] == pytest.approx(2 / 5) and long["outcome_status"] == "partial"
    assert long["matched_count"] == 2 and long["expected_count"] == 5 and not long["passed"]
    assert long["status"] == "completed", "a wrong answer is a measurement, not an error"
    curve = length_curve(rows)
    assert [point["target_input_tokens"] for point in curve] == [1024, 2048]
    assert curve[0]["mean_score"] == 1.0 and curve[1]["mean_score"] == pytest.approx(2 / 5)
    verdict = effective_length(rows)
    assert verdict["effective_length_tokens"] == 1024 and verdict["context_verified"]
    assert "fell below" in verdict["reason"]


def test_every_declared_task_yields_a_row_when_the_transport_fails_midway(dataset_root):
    adapter = RulerAdapter(tasks=("vt", "cwe"), lengths=(1024, 2048))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    transport = OracleTransport(items, fail_on=(2,))
    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(FakeCounter(), transport)))
    assert [row["task_id"] for row in rows] == list(adapter.task_ids(context))
    broken = [row for row in rows if row["status"] != "completed"]
    assert len(broken) == 1 and broken[0]["outcome_status"] == "transport_error"
    assert broken[0]["score"] == 0.0 and not broken[0]["passed"] and "dropped" in broken[0]["error"]
    assert broken[0]["target_input_tokens"] and broken[0]["actual_input_tokens"]
    assert len([row for row in rows if row["status"] == "completed"]) == 3
    assert not effective_length(rows)["curve"][0]["complete"]


def test_budget_exhaustion_marks_the_rest_instead_of_dropping_them(dataset_root):
    adapter = RulerAdapter(tasks=("vt", "cwe"), lengths=(1024, 2048))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    budget = iter([10_000.0, 10_000.0, 1.0, 1.0, 1.0])
    transport = OracleTransport(items)
    rows = adapter.run(make_context(dataset_root=dataset_root, remaining=lambda: next(budget),
                                    options=wired_options(FakeCounter(), transport)))
    assert [row["task_id"] for row in rows] == list(adapter.task_ids(context))
    assert [row["status"] for row in rows] == ["completed", "completed", "timeout", "timeout"]
    for row in rows[2:]:
        assert row["score"] == 0.0 and not row["passed"] and not row["model_evaluated"]
        assert row["target_input_tokens"] == parse_task_id(row["task_id"])[1]
        assert "budget" in row["error"] or "needed" in row["error"]
    assert len(transport.calls) == 2
    assert effective_length(rows)["effective_length_tokens"] == 1024
    assert "not every declared item completed" in effective_length(rows)["reason"]


def test_an_unwired_context_keeps_the_whole_denominator(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    rows = adapter.run(make_context(dataset_root=dataset_root, options={}))
    assert len(rows) == 1 and rows[0]["status"] == "environment_error"
    assert "token_counter" in rows[0]["error"] and rows[0]["ruler_task"] == "vt"
    context = make_context(dataset_root=dataset_root, options=wired_options(FakeCounter(), lambda p: {}))
    object.__setattr__(context, "artifacts", None)
    rows = adapter.run(context)
    assert rows[0]["status"] == "environment_error" and "evidence" in rows[0]["error"]


def test_a_deferred_task_is_named_not_silently_dropped(dataset_root):
    adapter = RulerAdapter()
    context = make_context(dataset_root=dataset_root, task_ids=("ruler/qa_2/1024",),
                           options=wired_options(FakeCounter(), lambda payload: {}))
    rows = adapter.run(context)
    assert len(rows) == 1 and rows[0]["status"] == "environment_error"
    assert rows[0]["outcome_status"] == "task_not_vendored" and "HotpotQA" in rows[0]["error"]


def test_a_length_that_misses_its_target_is_never_sent(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    transport = OracleTransport({})
    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(FakeCounter(), transport,
                                                          length_tolerance_tokens=1)))
    assert transport.calls == []
    assert rows[0]["status"] == "environment_error" and rows[0]["outcome_status"] == "length_target_unmet"
    assert rows[0]["actual_input_tokens"] < 1024 and rows[0]["length_delta_tokens"] < 0
    assert not rows[0]["length_within_tolerance"] and "tolerance" in rows[0]["error"]


def test_a_length_that_cannot_fit_the_served_window_is_refused(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(4096,))
    transport = OracleTransport({})
    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(FakeCounter(), transport, context_capacity=4096)))
    assert transport.calls == [] and rows[0]["outcome_status"] == "length_exceeds_context_window"
    assert rows[0]["status"] == "environment_error" and rows[0]["context_capacity"] == 4096


@pytest.mark.parametrize("delta,truncation,outcome,score,claim", [
    (0, False, "passed", 1.0, True),
    (2, False, "passed_context_unverified", 1.0, False),
    (-500, False, "invalid_context", 0.0, False),
    (0, True, "invalid_context", 0.0, False),
    (0, None, "passed", 1.0, False),
])
def test_context_accounting_records_both_axes_without_fabricating_a_claim(
        dataset_root, delta, truncation, outcome, score, claim):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    transport = OracleTransport(items, token_delta=delta, truncation=truncation)
    row = adapter.run(make_context(dataset_root=dataset_root,
                                   options=wired_options(FakeCounter(), transport)))[0]
    assert row["status"] == "completed" and row["outcome_status"] == outcome
    assert row["score"] == score and row["eligible_for_context_claim"] is claim
    # The retrieval axis is always recorded, even when the context axis destroyed the outcome.
    assert row["score_before_context_check"] == 1.0 and row["string_match_all"] == 1.0
    assert row["outcome_correct"] and row["retrieval_fraction"] == 1.0
    assert row["context_accounting_mismatch"] is (delta != 0)
    assert row["context_accounting_only"] is (delta == 2)
    assert row["context"]["observed_input_tokens"] == row["actual_input_tokens"] + delta
    assert row["context"]["actual_context_verified"] is claim


def test_an_estimated_counter_can_never_produce_a_context_claim(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    options = wired_options(FakeCounter(), OracleTransport(items), tokenizer_verified=False)
    row = adapter.run(make_context(dataset_root=dataset_root, options=options))[0]
    assert row["passed"] and row["score"] == 1.0
    assert not row["eligible_for_context_claim"]
    assert row["context"]["verification_status"] == "estimated_tokenizer"
    assert not effective_length([row])["context_verified"]


def test_a_mismatched_served_model_is_an_error_not_a_score(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    transport = OracleTransport(items, served_model="some-other-model")
    row = adapter.run(make_context(dataset_root=dataset_root,
                                   options=wired_options(FakeCounter(), transport)))[0]
    assert row["status"] == "environment_error" and row["outcome_status"] == "served_model_mismatch"
    assert row["score"] == 0.0 and "some-other-model" in row["error"]


def test_a_response_without_content_is_invalid_output(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    transport = OracleTransport(items, answer=lambda item: None)
    row = adapter.run(make_context(dataset_root=dataset_root,
                                   options=wired_options(FakeCounter(), transport)))[0]
    assert row["status"] == "invalid_output" and row["outcome_status"] == "no_assistant_content"
    assert row["score"] == 0.0 and row["model_evaluated"]


def test_a_defect_in_one_item_becomes_one_row_not_a_lost_denominator(dataset_root):
    adapter = RulerAdapter(tasks=("vt", "cwe"), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    inner = FakeCounter()

    def counter(messages, tools):  # a defect that only affects the variable-tracking item
        if "Memorize and track the chain" in messages[0]["content"]:
            raise ZeroDivisionError("a defect nobody anticipated")
        return inner(messages, tools)

    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(counter, OracleTransport(items))))
    assert [row["task_id"] for row in rows] == list(adapter.task_ids(context))
    assert {row["status"] for row in rows} == {"completed", "environment_error"}
    failed = [row for row in rows if row["status"] == "environment_error"][0]
    assert failed["ruler_task"] == "vt" and failed["outcome_status"] == "generator_failed"
    assert "ZeroDivisionError" in failed["error"] and failed["score"] == 0.0
    # A defect outside the generator and the transport is caught the same way.
    rows = adapter.run(make_context(dataset_root=dataset_root,
                                    options=wired_options(FakeCounter(), OracleTransport(items),
                                                          length_tolerance_tokens=0)))
    assert [row["outcome_status"] for row in rows] == ["adapter_error", "adapter_error"]
    assert all("length_tolerance_tokens" in row["error"] for row in rows)


def test_evidence_that_cannot_be_persisted_aborts_the_run(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    artifacts = RecordingArtifacts(fail_on="response.json")
    with pytest.raises(BenchmarkAborted, match="persisted"):
        adapter.run(make_context(dataset_root=dataset_root, artifacts=artifacts,
                                 options=wired_options(FakeCounter(), OracleTransport(items))))


def test_an_async_completion_route_is_awaited(dataset_root):
    adapter = RulerAdapter(tasks=("vt",), lengths=(1024,))
    context = make_context(dataset_root=dataset_root)
    items = build_items(adapter, context, FakeCounter())
    inner = OracleTransport(items)

    async def completion(payload):
        return inner(payload)

    row = adapter.run(make_context(dataset_root=dataset_root,
                                   options=wired_options(FakeCounter(), completion)))[0]
    assert row["outcome_status"] == "passed" and len(inner.calls) == 1


def test_budget_estimate_grows_with_length_and_rejects_nonsense():
    options = {}
    assert estimated_item_seconds(131072, 128, options) > estimated_item_seconds(4096, 128, options)
    assert estimated_item_seconds(131072, 128, options) > 60  # a 128K item is minutes, not seconds
    with pytest.raises(ValueError):
        estimated_item_seconds(4096, 128, {"prefill_tokens_per_second": 0})


def test_effective_length_fails_closed_without_a_baseline():
    verdict = effective_length([{"suite": "ruler", "task_id": "ruler/vt/4096", "target_input_tokens": 4096,
                                 "status": "timeout", "score": 0.0}])
    assert verdict["effective_length_tokens"] is None and "baseline" in verdict["reason"]
    with pytest.raises(ValueError):
        effective_length([], threshold=0)
