"""Inspect adapter for the frozen RULER retrieval subset (no sandbox)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import GenerateConfig

from .scoring import score_output, summarize


def make_task(payload: list[dict], provenance: dict, options: dict, generate_config: "GenerateConfig"):
    """Import Inspect only when constructing the task inside the controller."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, mean, scorer
    from inspect_ai.solver import generate

    @scorer(metrics=[mean()])
    def ruler_recall():
        async def score(state, target):
            answer, recall = score_output(state.output, list(target))
            return Score(value=recall, answer=answer)

        return score

    return Task(
        name="local_ruler_retrieval",
        version="1",
        dataset=[
            Sample(
                id=record["id"],
                input=record["input"] + record["answer_prefix"],
                target=record["outputs"],
                metadata={
                    "task": record["task"],
                    "index": record["index"],
                    "input_tokens": record["input_tokens"],
                },
            )
            for record in payload
        ],
        solver=generate(),
        scorer=ruler_recall(),
        config=generate_config,
        metadata={
            "selection": provenance,
            "benchmark_scope": "RULER retrieval subset; not the full 13-task RULER score",
            "prompt_variant": "Official base template plus answer prefix in one user message; local essay corpus",
            "reference_length_policy": "Fixed cl100k_base input-token budget, including answer prefix; not model-native context length",
            "scoring_policy": "Final completion only; fractional case-insensitive substring reference recall; separate reasoning excluded",
        },
    )
