"""Local fixture evaluations and optional Inspect integration. Import is inert."""

from __future__ import annotations

import copy
import json
from typing import Any

from .retrieval import build_niah_case, score_niah
from .tools import (
    ToolEpisode, ToolExpectation, failed_tool_score, score_tool_response,
    strict_argument_fixture, summarize_tool_scores, tool_message,
)


def run_demo_suite(*, inject_failure: bool = False) -> dict[str, Any]:
    """CPU-only deterministic positive/negative checks, explicitly synthetic.

    ``inject_failure`` adds a required timeout plus a wrong-type tool answer. Its
    purpose is to verify that failed attempts remain in report denominators.
    No model, tokenizer, API, worker, or external benchmark is run.
    """
    fixture = strict_argument_fixture()
    args = copy.deepcopy(fixture.calls[0].arguments)
    if inject_failure:
        args["options"]["retries"] = "3"
    tools = [score_tool_response(tool_message("configure_job", args), fixture)]
    no_call = ToolExpectation("tools/no-call-v1", fixture.tools, expected_text="No tool is needed.")
    tools.append(score_tool_response({"role": "assistant", "content": "No tool is needed."}, no_call))
    if inject_failure:
        tools.append(failed_tool_score("tools/intentional-timeout", "timeout", "Synthetic timeout for denominator test"))
    rows = [item.to_dict() for item in tools]
    episode = ToolEpisode()
    episode.consume(tool_message("read_file", {"path": episode.path}, "read"))
    episode.consume(tool_message("apply_patch", {
        "path": episode.path, "expected_revision": 1, "content": episode.expected_content,
    }, "patch"))
    episode.consume(tool_message("run_tests", {"path": episode.path}, "test"))
    rows.append(episode.result())
    for index, mode in enumerate(("text", "tool")):
        case = build_niah_case(depths=(0.05, 0.5, 0.95), missing_indices=(1,), mode=mode, seed=42 + index)
        response = (json.dumps(case.expected) if mode == "text"
                    else tool_message("submit_retrieved_keys", {"values": case.expected}))
        rows.append(score_niah(case, response))
    for row in rows:
        row.update({"synthetic": True, "model_evaluated": False})
        row.setdefault("metrics", {"score": row["score"]})
    passed = sum(row["passed"] for row in rows)
    return {"synthetic": True, "model_evaluated": False, "suite_revision": "local-demo-v1",
            "cases": rows, "summary": {
                "attempted": len(rows), "passed": passed, "success_all_required": passed / len(rows),
                "tools": summarize_tool_scores(tools), "context_claims_verified": 0,
            }}


__all__ = ["run_demo_suite", "build_niah_case", "score_niah", "score_tool_response", "ToolEpisode"]
