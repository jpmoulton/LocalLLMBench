"""A long-context candidate must be able to FINISH, or its context can never be verified.

Replays the measured failure: at a 262144 allocation one filled request cost 227-242 s. The candidate ran a
warm-up and three speed requests, then six NIAH variants at the same fill, inside a fixed 1200 s evaluation bound.
All four models died at 1201-1202 s, no retrieval row existed, and the context claim was refused.
"""

import json
from pathlib import Path

import pytest

from llmbench.containers.config import read_run_config
from llmbench.containers.derive import candidate_wall_for_tiers, derive_session_config, session_budgets_for
from llmbench.containers.proposals import (LONG_REQUEST_SECONDS, Proposal, apply, filled_request_seconds,
                                           size_long_context_work)

from test_containers_derive import counting_hasher, write_models

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.json"
MEASURED_SECONDS_PER_FILLED_REQUEST = 242.0  # the slowest of the four models at 261,372 input tokens
NIAH = ["single-early", "single-middle", "single-late", "multi", "missing", "tool-multi"]


def raw_candidate(input_tokens: int) -> dict:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["requested_input_tokens"] = input_tokens
    raw["benchmarks"] = [item if item["benchmark_id"] != "niah" else {**item, "task_ids": list(NIAH)}
                         for item in raw["benchmarks"]]
    return raw


def filled_requests(raw: dict) -> int:
    niah = sum(len(item["task_ids"]) for item in raw["benchmarks"] if item["benchmark_id"] == "niah")
    return raw["speed"]["warmup_repetitions"] + raw["speed"]["repetitions"] + niah


def test_the_default_workload_could_not_finish_at_the_ceiling():
    before = raw_candidate(261_376)
    assert filled_requests(before) == 1 + 3 + 6
    assert filled_requests(before) * MEASURED_SECONDS_PER_FILLED_REQUEST > 1200  # 2420 s of work, 1200 s allowed


def test_a_ceiling_candidate_is_sized_so_its_measured_cost_fits_its_own_bounds():
    raw = raw_candidate(261_376)
    size_long_context_work(raw)
    niah = next(item for item in raw["benchmarks"] if item["benchmark_id"] == "niah")
    assert niah["task_ids"] == ["single-late", "multi"]        # the two best variants, in their original order
    assert raw["speed"]["repetitions"] == 2 and raw["speed"]["warmup_repetitions"] == 1
    bounds = raw["bounds"]
    assert filled_requests(raw) * MEASURED_SECONDS_PER_FILLED_REQUEST + 600 < bounds["evaluation_seconds"]
    assert bounds["request_timeout_seconds"] > 2 * MEASURED_SECONDS_PER_FILLED_REQUEST
    assert bounds["evaluation_seconds"] < bounds["candidate_wall_seconds"] <= 14400
    tools = next(item for item in raw["benchmarks"] if item["benchmark_id"] == "tool-probes")
    assert tools["task_ids"]                                    # short suites are not thinned: they cost nothing extra


@pytest.mark.parametrize("input_tokens", [2048, 4096, 32_000, 60_000])
def test_ordinary_contexts_are_left_exactly_as_they_were(input_tokens):
    assert filled_request_seconds(input_tokens) <= LONG_REQUEST_SECONDS
    raw = raw_candidate(input_tokens)
    before = json.dumps(raw, sort_keys=True)
    size_long_context_work(raw)
    assert json.dumps(raw, sort_keys=True) == before


def test_at_least_one_retrieval_prompt_always_survives_and_bounds_are_never_lowered():
    raw = raw_candidate(1_000_000)                              # far beyond anything measured
    raw["bounds"] = {"evaluation_seconds": 9000, "request_timeout_seconds": 5000}
    size_long_context_work(raw)
    niah = next(item for item in raw["benchmarks"] if item["benchmark_id"] == "niah")
    assert niah["task_ids"] == ["multi"]                        # three needle depths in the one prompt that is left
    assert raw["bounds"]["evaluation_seconds"] >= 9000 and raw["bounds"]["request_timeout_seconds"] >= 5000
    assert raw["bounds"]["candidate_wall_seconds"] <= 14400


def test_a_context_range_session_gives_its_ceiling_candidate_room_and_a_valid_config(tmp_path):
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    session = derive_session_config(path, base=read_run_config(EXAMPLE), hasher=counting_hasher([]),
                                    budget_seconds=14400, context_floor=4096)
    ceiling = max(session.search.ctx_tiers)
    assert ceiling == 262144 and session.budgets.candidate_wall_seconds > 1800
    candidate = apply(session, session.base, Proposal(family="context", changes={"ctx_tier": ceiling}))
    assert candidate.requested_input_tokens == 261_376 and candidate.speed.repetitions == 2
    assert candidate.bounds.candidate_wall_seconds <= session.budgets.candidate_wall_seconds  # the grant admits it
    niah = next(item for item in candidate.benchmarks if item.benchmark_id == "niah")
    assert 1 <= len(niah.task_ids) <= 2
    small = apply(session, session.base, Proposal(family="context", changes={"ctx_tier": min(session.search.ctx_tiers)}))
    assert small.speed.repetitions == session.base.speed.repetitions and small.bounds == session.base.bounds


def test_a_session_too_small_for_a_long_candidate_keeps_its_budgets_and_a_default_session_is_unchanged(tmp_path):
    base = read_run_config(EXAMPLE)
    tight = session_budgets_for(3000)
    assert candidate_wall_for_tiers(base, (4864, 262144), tight, ceiling=261_376) == tight
    roomy = session_budgets_for(14400)
    assert candidate_wall_for_tiers(base, (4864, 262144), roomy, ceiling=261_376).candidate_wall_seconds > 1800
    assert candidate_wall_for_tiers(base, (4864, 16896), roomy, ceiling=16_000) == roomy
    (path,) = write_models(tmp_path / "w", ("Q4_K_M",))
    default = derive_session_config(path, base=base, hasher=counting_hasher([]), budget_seconds=14400)
    assert default.budgets.candidate_wall_seconds == 1800       # no range asked for: nothing about it changes
