import base64
import copy
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.coding.acceptance import acceptance_summary, acceptance_verdict
from llmbench.coding.fixtures import fixtures
from llmbench.coding.sandbox import WorkerResult
from llmbench.safety import SessionLock


def example(fixture, variant):
    cases, traces = [], []
    for check in fixture.checks:
        case = {'case_id': check.case_id, 'status': 'passed' if variant == 'reference' else 'incorrect',
                'passed': variant == 'reference', 'compile_ok': True, 'cleanup_confirmed': True,
                'input_unchanged': True, 'value_matches': variant == 'reference',
                'observation': {'runner_error': None, 'before': [], 'after': [], 'value': []}}
        raw = {'status': 'completed', 'returncode': 0, 'stderr_base64': base64.b64encode(
            ('LLMBENCH_CANDIDATE_ENTERED:' + check.case_id + '\n').encode()).decode()}
        if variant == 'hang':
            if fixture.language == 'python':
                case.update(status='timeout', compile_ok=False)
                raw.update(status='timeout', returncode=None, result_read_attempts=2,
                           result_read_status='completed', result_read_returncode=1,
                           result_read_stderr_base64=base64.b64encode(
                               b'cat: /tmp/llmbench-result.json: No such file or directory\n').decode())
            else:
                case['observation']['runner_error'] = (
                    "TimeoutExpired: Command '['node', '/workspace/.llmbench-node.cjs', '/workspace/x']' timed out")
        cases.append(case)
        traces.append({'case_id': check.case_id, 'worker': raw,
                       'job_argv': ['docker', 'run', '--name', 'llmbench-' + 'a' * 32]})
    return {'sample': {'synthetic': False, 'abort_campaign': False, 'passed': variant == 'reference',
                       'required_checks': len(cases), 'attempted_checks': len(cases), 'cases': cases},
            'abort_campaign': False, 'raw_bytes': json.dumps({'traces': traces}).encode()}


@pytest.mark.parametrize('fixture', fixtures(), ids=lambda f: f.language)
@pytest.mark.parametrize('variant', ['reference', 'initial', 'hang'])
def test_actual_calibration_evidence_passes(fixture, variant):
    assert acceptance_verdict(fixture, variant, example(fixture, variant))['as_expected']


@pytest.mark.parametrize('status', ['environment-error', 'result-missing', 'execution-error',
                                   'protocol-error', 'timeout', 'output-limit'])
@pytest.mark.parametrize('variant', ['reference', 'initial', 'hang'])
def test_infrastructure_failures_never_count_as_negative_success(status, variant):
    fixture = fixtures()[0]
    result = example(fixture, variant)
    result['sample']['passed'] = False
    for case in result['sample']['cases']:
        case.update(passed=False, status=status)
    result['raw_bytes'] = b'{"traces":[]}'
    assert not acceptance_verdict(fixture, variant, result)['as_expected']


@pytest.mark.parametrize('mutation', ['cleanup-missing', 'cleanup-false', 'synthetic', 'origin-missing',
                                      'not-attempted', 'missing-case', 'abort'])
def test_incomplete_or_uncertain_execution_cannot_pass(mutation):
    fixture = fixtures()[0]
    result = example(fixture, 'initial')
    sample = result['sample']
    if mutation == 'cleanup-missing':
        sample['cases'][0].pop('cleanup_confirmed')
    elif mutation == 'cleanup-false':
        sample['cases'][0]['cleanup_confirmed'] = False
    elif mutation == 'synthetic':
        sample['synthetic'] = True
    elif mutation == 'origin-missing':
        sample.pop('synthetic')
    elif mutation == 'not-attempted':
        sample['attempted_checks'] -= 1
    elif mutation == 'missing-case':
        sample['cases'].pop()
    else:
        result['abort_campaign'] = True
    assert not acceptance_verdict(fixture, 'initial', result)['as_expected']


@pytest.mark.parametrize('raw_change', [{'result_read_status': 'timeout'}, {'result_read_attempts': 0},
                                      {'result_read_returncode': 126}, {'result_read_stderr_base64':
                                      base64.b64encode(b'OCI runtime exec failed').decode()}])
def test_python_hang_requires_completed_missing_file_reads(raw_change):
    fixture = fixtures()[0]
    result = example(fixture, 'hang')
    trace = json.loads(result['raw_bytes'])
    trace['traces'][0]['worker'].update(raw_change)
    result['raw_bytes'] = json.dumps(trace).encode()
    assert not acceptance_verdict(fixture, 'hang', result)['as_expected']


def test_typescript_compiler_timeout_is_not_candidate_hang_evidence():
    fixture = fixtures()[1]
    result = example(fixture, 'hang')
    result['sample']['cases'][0]['observation']['runner_error'] = "TimeoutExpired: Command '['tsc', '--strict']' timed out"
    assert not acceptance_verdict(fixture, 'hang', result)['as_expected']


def test_final_acceptance_requires_complete_run_and_successful_absence_queries():
    rows = [{'as_expected': True, 'abort_campaign': False, 'cases': [{'cleanup_confirmed': True}]}]
    kwargs = dict(expected_rows=1, image='test', absence_checks=[{'name': 'owned', 'verified_absent': True}])
    assert acceptance_summary(rows, **kwargs)['accepted']
    for change in ({'expected_rows': 2}, {'absence_checks': []}, {'error': 'failed'},
                   {'absence_checks': [{'name': 'owned', 'verified_absent': False}]}):
        assert not acceptance_summary(rows, **{**kwargs, **change})['accepted']
    rows[0]['cases'][0].pop('cleanup_confirmed')
    assert not acceptance_summary(rows, **kwargs)['accepted']


def test_live_script_stops_after_abort_and_persists_failure(tmp_path, monkeypatch):
    script = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/coding_worker_acceptance.py'))
    main = script['main']
    globals_ = main.__globals__
    fixture = fixtures()[0]
    calls = []
    def run_fixture(*args, **kwargs):
        calls.append(args)
        result = example(fixture, 'reference')
        result['abort_campaign'] = result['sample']['abort_campaign'] = True
        result['sample']['cases'][0]['cleanup_confirmed'] = False
        return result
    def query(*args, **kwargs):
        assert kwargs['timeout_seconds'] == 10 and kwargs['max_output_bytes'] == 4096
        return WorkerResult('completed', 1, stderr=b'daemon unavailable')
    for name, value in {'fixtures': lambda: (fixture,), 'verified_local_image_id': lambda _: 'test',
                        'SessionLock': SimpleNamespace(read=lambda _: SessionLock(allow_container_execution=True)),
                        'DockerWorker': lambda **_: SimpleNamespace(executor=SimpleNamespace(run=query)),
                        'run_coding_fixture': run_fixture}.items():
        monkeypatch.setitem(globals_, name, value)
    out = tmp_path / 'acceptance'
    monkeypatch.setattr(sys, 'argv', ['acceptance', '--iidfile', 'test', '--output', str(out)])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 1 and len(calls) == 1
    summary = json.loads((out / 'acceptance-summary.json').read_text())
    assert not summary['accepted'] and not summary['complete'] and summary['any_abort']
    assert not summary['absence_query_verified']
    assert len(list(out.glob('*.trace.json'))) == len(list(out.glob('*.result.json'))) == 1
    saved = copy.deepcopy(summary['rows'][0])
    assert not saved['as_expected'] and saved['reasons']


@pytest.mark.parametrize('fixture', fixtures(), ids=lambda f: f.language)
@pytest.mark.parametrize('marker', ['', 'LLMBENCH_CANDIDATE_ENTERED:wrong-case', 'prefix-LLMBENCH_CANDIDATE_ENTERED:tail'])
def test_hang_requires_exact_case_driver_entry_marker(fixture, marker):
    result = example(fixture, 'hang')
    trace = json.loads(result['raw_bytes'])
    trace['traces'][0]['worker']['stderr_base64'] = base64.b64encode(marker.encode()).decode()
    result['raw_bytes'] = json.dumps(trace).encode()
    assert not acceptance_verdict(fixture, 'hang', result)['as_expected']


@pytest.mark.parametrize('missing', ['before', 'after', 'value', 'input_unchanged', 'value_matches'])
def test_initial_needs_complete_normal_execution_and_actual_value_mismatch(missing):
    fixture = fixtures()[0]
    result = example(fixture, 'initial')
    for case in result['sample']['cases']:
        (case if missing in {'input_unchanged', 'value_matches'} else case['observation']).pop(missing)
    assert not acceptance_verdict(fixture, 'initial', result)['as_expected']
