"""BigCodeBench Complete task, sandbox scoring, and result interpretation."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Solver, TaskState, generate, solver
from inspect_ai.util import ExecResult, sandbox
from inspect_evals.bigcodebench.bigcodebench import find_code, get_record_to_sample


# This wrapper, and all supplied Python, run ONLY inside the sample's sandbox.
# Separate processes/directories prevent canonical globals and files reaching scoring.
SANDBOX_WRAPPER = """import os, sys, tempfile
with tempfile.TemporaryDirectory(prefix='bigcodebench-', dir='/tmp') as workdir:
    os.chdir(workdir)
    source = sys.stdin.read()
    sys.argv = ['bigcodebench.py']
    __file__ = os.path.join(workdir, 'bigcodebench.py')
    exec(compile(source, __file__, 'exec'), globals())
"""
TEST_MAIN = "\nif __name__ == '__main__':\n    unittest.main()\n"
TRUNCATION_REASONS = {"max_tokens", "model_length"}


async def execute_tests(code: str, tests: str, timeout: int) -> ExecResult[str]:
    result = await sandbox().exec(
        ["python", "-c", SANDBOX_WRAPPER],
        input=code + "\n" + tests + TEST_MAIN,
        cwd="/tmp",
        timeout=timeout,
        timeout_retry=False,
    )
    # Exit 1 is Python/unittest's normal incorrect-solution result. Docker/tool
    # failures, signals (including OOM), and allocation failures are not scores.
    if result.returncode not in (0, 1) or "MemoryError" in result.stderr:
        raise RuntimeError(
            f"Sandbox execution failed (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


@solver
def reference_preflight(test_timeout: int) -> Solver:
    async def solve(state: TaskState, generate: Any) -> TaskState:
        result = await execute_tests(state.target.text, state.metadata["test"], test_timeout)
        if not result.success:
            raise RuntimeError(
                "Canonical-solution preflight failed; generation was not attempted. "
                "This is a dataset/environment error, not an incorrect model answer.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return state

    return solve


@scorer(metrics=[mean()])
def sandbox_verify(test_timeout: int) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        if not state.output.choices:
            raise RuntimeError("Model returned no completion choices.")
        reason = state.output.choices[0].stop_reason
        if reason in TRUNCATION_REASONS:
            raise RuntimeError(f"Generation truncated ({reason}); no score assigned.")
        if state.output.error:
            raise RuntimeError(f"Model generation failed: {state.output.error}")
        if reason not in {"stop", "content_filter"}:
            raise RuntimeError(f"Unexpected generation finish reason: {reason}")
        # Inspect's OpenAI-compatible provider keeps reasoning_content in a
        # ContentReasoning block; completion contains only the final text.
        answer = find_code(state.output.completion)
        result = await execute_tests(answer, state.metadata["test"], test_timeout)
        return Score(
            value=CORRECT if result.success else INCORRECT,
            answer=answer,
            explanation=f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    return score


def sample_summary(sample: EvalSample, task_id: str) -> dict:
    output = sample.output
    reason = output.choices[0].stop_reason if output.choices else None
    values = {name: score.value for name, score in (sample.scores or {}).items()}
    truncated = reason in TRUNCATION_REASONS
    error = sample.error.model_dump(mode="json") if sample.error else None
    if truncated:
        status = "truncation"
    elif error or output.error or sample.limit:
        status = "error"
    elif list(values.values()) == [CORRECT]:
        status = "pass"
    elif list(values.values()) == [INCORRECT]:
        status = "fail"
    else:
        status = "error"
        error = {"message": "Sample has no unambiguous final score."}
    return {
        "id": task_id,
        "inspect_id": sample.id,
        "status": status,
        "finish_reason": reason,
        "finish_reason_format": "Inspect normalized stop reason (max_tokens corresponds to API length)",
        "token_usage": output.usage.model_dump(mode="json") if output.usage else None,
        "scores": values,
        "error": error,
        "model_error": output.error,
        "limit": sample.limit.model_dump(mode="json") if sample.limit else None,
    }


def summarize(logs: list[EvalLog], provenance: dict, run_error: dict | None) -> dict:
    selected_ids = provenance.get("selected_ids", [])
    by_id = {str(int(task_id.rsplit("/", 1)[1])): task_id for task_id in selected_ids}
    samples: dict[str, dict] = {}
    log_errors = []
    for log in logs:
        if log.status != "success" or log.error:
            log_errors.append({
                "log": log.location,
                "status": log.status,
                "error": log.error.model_dump(mode="json") if log.error else None,
            })
        for sample in log.samples or []:
            task_id = by_id.get(str(sample.id))
            if task_id is None or task_id in samples:
                log_errors.append({"message": f"Unexpected or duplicate sample ID: {sample.id}"})
                continue
            samples[task_id] = sample_summary(sample, task_id)
    ordered = [samples.get(task_id, {"id": task_id, "status": "missing"}) for task_id in selected_ids]
    counts = {status: sum(item["status"] == status for item in ordered)
              for status in ("pass", "fail", "error", "truncation", "missing")}
    complete = bool(selected_ids) and bool(logs) and not run_error and not log_errors and not any(
        counts[status] for status in ("error", "truncation", "missing")
    )
    return {
        "benchmark": "BigCodeBench",
        "prompt_variant": "Complete",
        "status": "completed" if complete else "incomplete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "denominator": len(selected_ids),
        "scored": counts["pass"] + counts["fail"],
        "counts": counts,
        "pass_rate": counts["pass"] / len(selected_ids) if complete else None,
        "pass_rate_policy": "Passes / all selected tasks, published only when every task is scored without errors or truncation.",
        "inspect_logs": [log.location for log in logs],
        "inspect_log_errors": log_errors,
        "run_error": run_error,
        "samples": ordered,
    }


def make_task(
    payload: list[dict],
    provenance: dict,
    options: dict,
    generate_config: GenerateConfig,
) -> Task:
    """Build the Complete task while preserving Inspect's numeric sample IDs."""
    convert = get_record_to_sample()
    return Task(
        name="local_bigcodebench_complete",
        version="v0.1.4",
        dataset=[convert(record) for record in payload],
        setup=reference_preflight(options["test_timeout"]),
        solver=generate(),
        scorer=sandbox_verify(options["test_timeout"]),
        sandbox=("docker", str(Path(__file__).with_name("sandbox.yaml"))),
        config=generate_config,
        metadata={"selection": provenance, "prompt_variant": "Complete"},
    )
