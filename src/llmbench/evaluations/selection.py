"""Turn benchmark selections into concrete Inspect selections and generated NIAH cases.

Same construction as ``live.LiveExecutor`` (which stays untouched), with the token counter
supplied by whichever engine is attached. Import is inert; the counter is only called here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

from ..config import GenerationSettings, TaskSelection
from .retrieval import NIAH_VARIANTS
from .retrieval import NiahCase, build_niah_case

ChatCounter = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
SKIPPED_SUITES = frozenset({"coding"})  # never an Inspect task; needs the worker broker


def serving_tool_payload(tools: Iterable[Mapping[str, Any]], *, strict_tools: bool) -> list[dict[str, Any]]:
    """Re-serialise fixture tool schemas exactly as the serving path will send them (LIVE-008).

    A NIAH tool case is counted here but sent by ``evaluations.capture``, which rebuilds every tool from
    Inspect's ``ToolInfo``/``ToolParams`` models (``inspect_tasks._tool_infos``) and appends ``strict``.
    Pydantic emits its own field-declaration order, not the fixture's authoring order, and llama.cpp keeps
    the request's key order when the chat template renders ``{{ tool | tojson }}``. Counting the authoring
    order therefore counts a prompt the server never renders: in the first live campaign the two orders of
    the identical schema differed by exactly one token, which invalidated every candidate's context claim.
    Inspect's own models are used here so the field order is never duplicated by hand; a schema Inspect
    refuses raises instead of being counted in a shape that will not be sent.
    """
    if type(strict_tools) is not bool:
        raise ValueError("strict_tools must be a boolean")
    from inspect_ai.tool import ToolInfo, ToolParams  # runtime-only import: module import stays inert

    payload = []
    for tool in tools:
        function = dict(tool)["function"]
        info = ToolInfo(name=function["name"], description=function["description"],
                        parameters=ToolParams.model_validate(function["parameters"]))
        payload.append({"type": "function", "function": {
            "name": info.name, "description": info.description,
            "parameters": info.parameters.model_dump(exclude_none=True, by_alias=True),
            "strict": strict_tools,
        }})
    return payload


def niah_task_id(name: str, split: str, seed: int) -> str:
    """The named variant is the paired task unit; the payload hash records generated content."""
    return f"niah/{name}/{split}/seed-{seed}"


def as_task_selection(item: Any) -> TaskSelection:
    if isinstance(item, TaskSelection):
        return item
    convert = getattr(item, "to_task_selection", None)
    return convert() if callable(convert) else TaskSelection.model_validate(item)


def _category_of(suite: str) -> str:
    """The category a denominator row must claim, taken from the registry so a backfilled row can
    never disagree with the category the adapter's own rows report (a mismatch would split one
    benchmark across two quality buckets in analysis). Falls back to the historic default only for
    a suite the registry does not list."""
    from ..registry import builtin_registry
    entry = builtin_registry()._by_id.get(suite)
    return entry.category if entry is not None else "tools"


def expected_task_rows(benchmarks: Iterable[Any]) -> list[dict[str, Any]]:
    """Denominator rows, computable before any server contact."""
    rows = []
    for selection in map(as_task_selection, benchmarks):
        for task_id in selection.task_ids:
            rows.append({
                "task_id": niah_task_id(task_id, selection.split, selection.fixture_seed)
                if selection.suite == "niah" else task_id,
                "category": _category_of(selection.suite),
                "suite": selection.suite, "suite_revision": selection.revision, "split": selection.split,
                "fixture_seed": selection.fixture_seed,
            })
    return rows


def build_selected_cases(benchmarks: Iterable[Any], *, requested_input_tokens: int, ctx_size: int,
                         generation: GenerationSettings, counter: ChatCounter, tokenizer_id: str,
                         template_id: str, tokenizer_verified: bool = False,
                         ) -> tuple[list[NiahCase], list[TaskSelection]]:
    """``counter(messages, tools)`` returns ``{"tokens": int, ...}`` from the serving template.

    ``tokenizer_verified`` may be true only when that counter was shown to equal the serving
    route's own prompt count; otherwise every case is honestly labelled "estimated".

    The counter is handed the tool schema in the exact serialisation the capture model will send
    (``serving_tool_payload``), never the fixture's authoring order: LIVE-008 showed that counting a
    differently ordered rendering of the same schema silently costs a token on the tool path.
    """
    if type(tokenizer_verified) is not bool:
        raise ValueError("tokenizer_verified must be a boolean")

    def count(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
        served = serving_tool_payload(tools, strict_tools=generation.strict_tools) if tools else []
        return counter(messages, served)["tokens"]

    cases: list[NiahCase] = []
    selections: list[TaskSelection] = []
    for selection in map(as_task_selection, benchmarks):
        if selection.suite in SKIPPED_SUITES:
            continue
        if selection.suite != "niah":
            selections.append(selection)
            continue
        if selection.revision != "local-niah-v1" or set(selection.task_ids) - set(NIAH_VARIANTS):
            raise ValueError("NIAH selection requires a supported named variant and local-niah-v1")
        selected = []
        for name in selection.task_ids:
            case = build_niah_case(
                target_input_tokens=requested_input_tokens, context_capacity=ctx_size,
                reserved_output_tokens=generation.max_output_tokens, seed=selection.fixture_seed,
                split=selection.split, token_counter=count,
                tokenizer_verified=tokenizer_verified, tokenizer_id=tokenizer_id, template_id=template_id,
                **NIAH_VARIANTS[name])
            case = replace(case, task_id=niah_task_id(name, selection.split, selection.fixture_seed))
            cases.append(case)
            selected.append(case.task_id)
        selections.append(TaskSelection(suite="niah", revision=selection.revision, task_ids=tuple(selected),
                                        split=selection.split, fixture_seed=selection.fixture_seed))
    return cases, selections
