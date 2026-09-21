import hashlib
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from llmbench.coding.adapters import (
    CaseResult, ResultEnvelope, UpstreamPin, authorize_command, evaluation_command, import_result,
)
from llmbench.config import RunMode
from llmbench.safety import OperationForbidden, SessionLock


def pin(suite="tool-eval-bench"):
    return UpstreamPin(suite=suite, source_commit="a" * 40, dataset_sha256="b" * 64,
                       scorer_commit="c" * 40, case_ids=("TC-01", "TC-02", "TC-03"),
                       languages=("python",), subset_label="private development subset",
                       environment_sha256="d" * 64)


def test_all_required_denominator_includes_environment_failure(tmp_path):
    raw = b'{"final_score":100,"completion_rate":0.6666667}'
    tmp_path.joinpath("raw.json").write_bytes(raw)
    envelope = ResultEnvelope(pin=pin(), run_id="run-1", model_configuration_sha256="e" * 64,
                              raw_artifact="raw.json", raw_sha256=hashlib.sha256(raw).hexdigest(),
                              official_metrics_json='{"final_score":100}', source_format="declared-normalized-v1",
                              cases=(CaseResult(case_id="TC-01", status="passed", eligible_for_upstream_score=True,
                                                first_attempt_success=True, duration_seconds=1.0),
                                     CaseResult(case_id="TC-02", status="passed", eligible_for_upstream_score=True,
                                                duration_seconds=2.0, repairs=1),
                                     CaseResult(case_id="TC-03", status="timeout", eligible_for_upstream_score=False,
                                                duration_seconds=30.0)))
    result = import_result(envelope, artifact_root=tmp_path)
    assert result.attempted == 3 and result.completed == 2
    assert result.all_required_success_rate == 2 / 3
    assert json.loads(result.official_metrics_json) == {"final_score": 100}
    assert result.upstream_eligibility_ids == ("TC-01", "TC-02")
    assert result.artifact_verified and not result.normalization_verified
    tmp_path.joinpath("raw.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        import_result(envelope, artifact_root=tmp_path)


def test_pin_and_cases_reject_ambiguous_identity():
    data = pin().model_dump()
    data["source_commit"] = "main"
    with pytest.raises(ValidationError):
        UpstreamPin.model_validate(data)
    with pytest.raises(ValidationError, match="First-attempt"):
        CaseResult(case_id="x", status="failed", eligible_for_upstream_score=True,
                   first_attempt_success=True, duration_seconds=1.0)


def test_documented_commands_have_explicit_endpoint_ids_seed_and_isolation():
    command = evaluation_command(pin(), base_url="http://127.0.0.1:1234", model="already-loaded")
    assert "--scenarios" in command.argv and "TC-03" in command.argv
    assert command.environment == (("TOOL_EVAL_MODEL", "already-loaded"),)
    assert command.requires_inference
    with pytest.raises(OperationForbidden):
        authorize_command(command, session_lock=SessionLock(), mode=RunMode.LIVE)
    with pytest.raises(OperationForbidden):
        authorize_command(replace(command, requires_inference=False), session_lock=SessionLock(), mode=RunMode.LIVE)
    evaluation = evaluation_command(pin("evalplus"))
    assert evaluation.argv == ("evalplus.evaluate", "--dataset", "humaneval", "--samples", "/workspace/samples.jsonl")
    assert evaluation.requires_isolation and not evaluation.requires_inference
    with pytest.raises(OperationForbidden):
        authorize_command(evaluation, session_lock=SessionLock(), mode=RunMode.LIVE)
    with pytest.raises(ValueError, match="explicit local"):
        evaluation_command(pin(), model="x")


@pytest.mark.parametrize("suite", ["aider-polyglot", "bfcl", "ruler"])
def test_unverified_execution_is_explicitly_unsupported(suite):
    with pytest.raises(NotImplementedError, match="import-only"):
        evaluation_command(pin(suite))

