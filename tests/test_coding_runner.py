import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmbench.coding.fixtures import fixtures
from llmbench.coding.runner import run_coding_fixture, validate_observation
from llmbench.coding.sandbox import DockerWorker, WorkerResult
from llmbench.config import RunMode
from llmbench.safety import OperationForbidden, SessionLock


IMAGE = 'local/test@sha256:' + 'a' * 64


class FixtureExecutor:
    def __init__(self, fixture, *, mutate=False, timeout_first=False, no_result=False, cleanup_absent=True,
                 absent_polls=0, result_override=None):
        self.fixture, self.mutate, self.timeout_first = fixture, mutate, timeout_first
        self.no_result, self.cleanup_absent = no_result, cleanup_absent
        self.absent_polls, self.result_override, self.polls = absent_polls, result_override, {}
        self.calls = []
        self.payloads = {}
        self.running = {}

    def run(self, argv, **limits):
        self.calls.append((argv, limits))
        if argv[1] == 'run':
            assert '--detach' in argv  # Returning after PID 1 exit would destroy tmpfs.
            name = argv[argv.index('--name') + 1]
            mount = argv[argv.index('--mount') + 1]
            path = Path(mount.removeprefix('type=bind,source=').removesuffix(',target=/workspace,readonly'))
            config = json.loads(path.joinpath('.llmbench-input.json').read_text())
            check = next(case for case in self.fixture.checks if case.case_id == config['case_id'])
            before = json.loads(check.arguments_json)
            observed = {'protocol': 1, 'case_id': check.case_id, 'before': before,
                        'after': [] if self.mutate else before, 'compile_ok': True,
                        'runner_error': None, 'value': json.loads(check.expected_json)}
            self.payloads[name] = json.dumps(observed).encode()
            self.running[name] = not self.no_result
            if self.timeout_first:
                self.timeout_first = False
                return WorkerResult('timeout', None)
            return WorkerResult('completed', 0, stdout=b'owned-container-id')
        if argv[1] == 'exec':
            # Lifecycle-aware `docker exec <name> cat <result>`: docker cp cannot read tmpfs.
            assert argv[3:] == ('cat', '/tmp/llmbench-result.json') and len(argv) == 5
            name = argv[2]
            if not self.running.get(name):  # exec into a stopped/removed container is a daemon error
                return WorkerResult('completed', 1, stderr=b'Error response from daemon: container is not running')
            self.polls[name] = self.polls.get(name, 0) + 1
            if self.polls[name] <= self.absent_polls:  # driver has not renamed the result into place yet
                return WorkerResult('completed', 1, stderr=b'cat: /tmp/llmbench-result.json: No such file or directory')
            return WorkerResult('completed', 0, stdout=self.result_override or self.payloads[name])
        if argv[1] == 'inspect':
            return WorkerResult('completed', 0, stdout=b'true' if self.running.get(argv[-1]) else b'false')
        if argv[1] == 'logs':
            return WorkerResult('completed', 0, stdout=b'{"passed":true,"score":1}')
        if argv[1] == 'rm':
            if self.cleanup_absent:
                self.running[argv[-1]] = False
                self.payloads.pop(argv[-1], None)  # tmpfs disappears at stop/removal.
            # A failed removal alone is not cleanup failure if absence is verified.
            return WorkerResult('completed', 1, stderr=b'No such container')
        if argv[1] == 'ps':
            return WorkerResult('completed', 0, stdout=b'' if self.cleanup_absent else b'still-running')
        raise AssertionError(argv)


def worker_for(tmp_path, fixture, **options):
    executor = FixtureExecutor(fixture, **options)
    worker = DockerWorker(allowed_root=tmp_path / 'candidates',
                          session_lock=SessionLock(allow_container_execution=True), mode=RunMode.LIVE,
                          executor=executor)
    return worker, executor


@pytest.mark.parametrize('fixture', fixtures(), ids=lambda f: f.language)
def test_production_orchestration_with_fake_container_observations(tmp_path, fixture):
    worker, executor = worker_for(tmp_path, fixture)
    patch = {item.path: item.content for item in fixture.reference_files}
    result = run_coding_fixture(fixture, patch, worker=worker, image=IMAGE)
    sample = result['sample']
    assert sample['score'] == 1. and sample['passed_checks'] == len(fixture.checks)
    assert sample['required_checks'] == sample['attempted_checks'] == len(fixture.checks)
    assert sample['synthetic'] is True  # injected transport cannot claim real execution
    assert all(case['input_unchanged'] for case in sample['cases'])
    roots = sorted(Path(result['candidate_path']).iterdir())
    assert len(roots) == len(fixture.checks)
    for directory in roots:
        for source in fixture.reference_files:
            assert directory.joinpath(source.path).read_text() == source.content
        inputs = json.loads(directory.joinpath('.llmbench-input.json').read_text())
        assert 'expected_json' not in inputs and 'expected' not in inputs
    trace = json.loads(result['raw_bytes'])
    assert base64.b64decode(trace['traces'][0]['worker']['stdout_base64']) == b'{"passed":true,"score":1}'
    assert sum(argv[1] == 'run' for argv, _ in executor.calls) == len(fixture.checks)


def test_input_mutation_fails_even_when_return_value_is_correct(tmp_path):
    fixture = fixtures()[0]
    worker, _ = worker_for(tmp_path, fixture, mutate=True)
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 0. and sample['passed_checks'] == 0


def test_stdout_success_cannot_replace_a_missing_result_file(tmp_path):
    fixture = fixtures()[0]
    worker, _ = worker_for(tmp_path, fixture, no_result=True)
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 0.
    assert all(case['status'] == 'result-missing' for case in sample['cases'])


def test_timeout_retains_denominator_and_other_cases(tmp_path):
    fixture = fixtures()[0]
    worker, _ = worker_for(tmp_path, fixture, timeout_first=True)
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 0.
    assert sample['required_checks'] == 4 and sample['passed_checks'] == 3
    assert sample['cases'][0]['status'] == 'timeout'


def test_unverified_cleanup_halts_remaining_cases(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture, cleanup_absent=False)
    result = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)
    sample = result['sample']
    assert sample['required_checks'] == 4 and sample['attempted_checks'] == 1
    assert sample['score'] == 0.
    assert result['abort_campaign'] is True and sample['abort_campaign'] is True
    assert result['raw_bytes']
    assert sum(argv[1] == 'run' for argv, _ in executor.calls) == 1


def test_denial_occurs_before_staging_or_process(tmp_path):
    fixture = fixtures()[0]
    worker = DockerWorker(allowed_root=tmp_path / 'new', session_lock=SessionLock())
    with pytest.raises(OperationForbidden):
        run_coding_fixture(fixture, None, worker=worker, image=IMAGE)
    assert not (tmp_path / 'new').exists()


def test_result_protocol_rejects_duplicate_keys_and_false_case_id():
    check = fixtures()[0].checks[0]
    assert not validate_observation(b'{"protocol":1,"protocol":1}', check=check)['passed']
    assert not validate_observation(b'{"protocol":1,"case_id":"other"}', check=check)['passed']


def test_total_fixture_budget_keeps_unattempted_checks_as_failures(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture)
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE, timeout_seconds=.01)['sample']
    assert sample['required_checks'] == 4 and sample['attempted_checks'] == 0
    assert sample['score'] == 0.
    assert not executor.calls


def test_trusted_driver_sources_compile_without_executing_candidates():
    from llmbench.coding.drivers import PYTHON_DRIVER, NODE_DRIVER
    compile(PYTHON_DRIVER, 'trusted-container-driver', 'exec')
    assert 'expected_json' not in PYTHON_DRIVER and 'expected_json' not in NODE_DRIVER
    assert "'/tmp/llmbench-result.json'" in NODE_DRIVER
    assert 'sleep(30)' in PYTHON_DRIVER
    assert 'publish_result(result_path)' in PYTHON_DRIVER
    assert "rename('/tmp/llmbench-result.pending', '/tmp/llmbench-result.json')" in NODE_DRIVER


def test_tmpfs_result_is_collected_before_container_stops(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture)
    result = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)
    assert result['sample']['score'] == 1.
    reads = 0
    for index, (argv, limits) in enumerate(executor.calls):
        assert argv[1] != 'cp'  # docker cp cannot read the tmpfs result.
        if argv[1] == 'exec':
            reads += 1
            name = argv[2]
            assert argv == ('docker', 'exec', name, 'cat', '/tmp/llmbench-result.json')
            assert limits['timeout_seconds'] <= 10
            assert any(previous[0][1] == 'run' and name in previous[0] for previous in executor.calls[:index])
            assert not any(previous[0][1] == 'rm' and name in previous[0] for previous in executor.calls[:index])
    assert reads == len(fixture.checks)  # collected as soon as published; no keepalive wait
    assert not any(executor.running.values())
    assert executor.payloads == {}


def test_result_read_polls_until_the_driver_publishes(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture, absent_polls=3)
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 1.
    kinds = [argv[1] for argv, _ in executor.calls]
    assert kinds.count('exec') == 4 * len(fixture.checks)
    assert kinds.count('inspect') == 3 * len(fixture.checks)  # liveness is checked after every absent read
    assert kinds.count('rm') == kinds.count('ps') == len(fixture.checks)


@pytest.mark.parametrize('payload', [b' ' * (1_048_576 + 1), b'not json'], ids=['oversize', 'garbage'])
def test_invalid_raw_result_bytes_fail_the_case_and_are_cleaned_up(tmp_path, payload):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture, result_override=payload)
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 0. and sample['passed_checks'] == 0
    assert sample['attempted_checks'] == len(fixture.checks)
    if len(payload) > 1_048_576:
        assert all(case['status'] == 'protocol-error' for case in sample['cases'])
    assert all(case['cleanup_confirmed'] for case in sample['cases'])


def test_empty_result_read_is_a_protocol_error(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture)
    original = executor.run
    def empty_read(argv, **limits):
        outcome = original(argv, **limits)
        return WorkerResult('completed', 0) if argv[1] == 'exec' else outcome
    executor.run = empty_read
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert all(case['status'] == 'protocol-error' and case['cleanup_confirmed'] for case in sample['cases'])


@pytest.mark.parametrize('status', ['timeout', 'output-limit', 'pipe-error'])
def test_unfinished_result_read_propagates_and_still_cleans_up(tmp_path, status):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture)
    original = executor.run
    def broken_read(argv, **limits):
        if argv[1] == 'exec':
            executor.calls.append((argv, limits))
            return WorkerResult(status, None, stdout=b'x' * 10)
        return original(argv, **limits)
    executor.run = broken_read
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 0.
    assert all(case['status'] == status and case['cleanup_confirmed'] for case in sample['cases'])
    kinds = [argv[1] for argv, _ in executor.calls]
    assert kinds.count('exec') == kinds.count('rm') == len(fixture.checks)  # no retry, exact cleanup


def test_hung_candidate_ends_as_bounded_timeout_with_cleanup(tmp_path, monkeypatch):
    from llmbench.coding import sandbox
    fixture = fixtures()[0]
    # The container stays alive but never publishes: every read fails until the deadline.
    worker, executor = worker_for(tmp_path, fixture, absent_polls=10 ** 9)
    now = [0.]
    def tick():
        now[0] += .5
        return now[0]
    # Replace only the sandbox module's view of time, never the global module.
    monkeypatch.setattr(sandbox, 'time', SimpleNamespace(monotonic=tick, sleep=lambda seconds: None))
    sample = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)['sample']
    assert sample['score'] == 0.
    assert all(case['status'] == 'timeout' and case['cleanup_confirmed'] for case in sample['cases'])
    kinds = [argv[1] for argv, _ in executor.calls]
    assert 0 < kinds.count('exec') <= 30 * len(fixture.checks)
    assert kinds.count('rm') == kinds.count('ps') == len(fixture.checks)
    assert not any(executor.running.values())


def test_cleanup_exception_preserves_raw_results_and_campaign_abort(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture)
    original = executor.run
    def fail_cleanup(argv, **limits):
        if argv[1] == 'rm':
            raise RuntimeError('daemon connection lost during cleanup')
        return original(argv, **limits)
    executor.run = fail_cleanup
    result = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)
    assert result['abort_campaign'] is True
    assert result['sample']['required_checks'] == 4
    assert result['sample']['attempted_checks'] == 1
    assert result['sample']['cases'][0]['status'] == 'cleanup-unverified'
    assert b'daemon connection lost' in result['raw_bytes']



def test_failed_read_bytes_are_preserved_in_durable_fixture_trace(tmp_path):
    fixture = fixtures()[0]
    worker, executor = worker_for(tmp_path, fixture)
    original = executor.run
    def fail_read(argv, **limits):
        if argv[1] == 'exec':
            return WorkerResult('timeout', None, b'partial-result-marker', b'failed-read-marker')
        return original(argv, **limits)
    executor.run = fail_read
    result = run_coding_fixture(fixture, None, worker=worker, image=IMAGE)
    raw = json.loads(result['raw_bytes'])
    for trace in raw['traces']:
        row = trace['worker']
        assert base64.b64decode(row['result_read_stdout_base64']) == b'partial-result-marker'
        assert base64.b64decode(row['result_read_stderr_base64']) == b'failed-read-marker'
        assert row['result_read_status'] == 'timeout' and row['result_read_attempts'] == 1
    assert all(case['cleanup_confirmed'] for case in result['sample']['cases'])
