import json

import pytest

from llmbench.coding.fixtures import fixtures, score_observations, write_candidate


def test_fixture_languages_are_real_and_repeatable():
    first = fixtures()
    assert {f.language for f in first} == {"python", "typescript", "javascript"}
    assert [f.identity() for f in first] == [f.identity() for f in fixtures()]
    ts = next(f for f in first if f.language == "typescript")
    assert any(f.path.endswith(".ts") for f in ts.initial_files)
    assert ts.reference_files != ts.initial_files


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda f: f.fixture_id)
def test_evaluator_checks_observations_not_claimed_success(fixture):
    good = {case.case_id: json.loads(case.expected_json) for case in fixture.checks}
    assert score_observations(fixture, good, compile_ok=True, execution_ok=True).success
    bad = dict(good)
    bad[fixture.checks[0].case_id] = {"message": "all tests passed"}
    result = score_observations(fixture, bad, compile_ok=True, execution_ok=True)
    assert not result.success
    assert result.passed == len(fixture.checks) - 1
    assert not score_observations(fixture, good, compile_ok=False, execution_ok=True).success
    assert not score_observations(fixture, good, compile_ok=True, execution_ok=False).success
    missing = score_observations(fixture, {}, compile_ok=True, execution_ok=True)
    assert missing.required == len(fixture.checks)
    assert missing.completed == missing.passed == 0


def test_workspace_only_contains_candidate_sources(tmp_path):
    fixture = fixtures()[0]
    candidate = write_candidate(tmp_path / "new", fixture)
    assert {p.name for p in candidate.iterdir()} == {"batches.py"}
    assert candidate.joinpath("batches.py").read_text() == fixture.initial_files[0].content
    assert "raise ValueError" not in candidate.joinpath("batches.py").read_text()
    with pytest.raises(ValueError, match="overwrite"):
        write_candidate(candidate, fixture)
    with pytest.raises(ValueError, match="outside"):
        write_candidate(tmp_path / "escape", fixture, {"../evaluator.py": "anything"})
    with pytest.raises(ValueError, match="budget"):
        write_candidate(tmp_path / "large", fixture, {"batches.py": "x" * 100}, max_bytes=10)


def test_boolean_does_not_equal_numeric_expected():
    from llmbench.coding.fixtures import CheckCase, CodingFixture
    f = CodingFixture("strict", "python", "p", "x:f", (), (), (CheckCase("x", "[]", "1"),))
    assert not score_observations(f, {"x": True}, compile_ok=True, execution_ok=True).success

