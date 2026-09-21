import io
import subprocess

import pytest

from llmbench.coding.sandbox import BoundedProcessExecutor, DockerWorker
from llmbench.config import RunMode
from llmbench.safety import OperationForbidden, SessionLock


LOCK = SessionLock(allow_container_execution=True)
NAME = 'llmbench-' + '1' * 32


class FakeProcess:
    def __init__(self, *, stdout=b'hello', stderr=b'error', running=False):
        self.stdout, self.stderr = io.BytesIO(stdout), io.BytesIO(stderr)
        self.running = running
        self.killed = False
        self.returncode = None if running else 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired('fake', timeout)
        return self.returncode


def test_popen_is_guarded_and_has_no_shell_or_stdin():
    process, calls = FakeProcess(), []
    def factory(argv, **kwargs):
        calls.append((argv, kwargs))
        return process
    denied = BoundedProcessExecutor(popen_factory=factory)
    with pytest.raises(OperationForbidden):
        denied.run(('docker', 'rm', '-f', NAME), timeout_seconds=1, max_output_bytes=100)
    assert calls == []
    allowed = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE, popen_factory=factory)
    result = allowed.run(('docker', 'rm', '-f', NAME), timeout_seconds=1, max_output_bytes=100)
    assert result.stdout == b'hello' and result.stderr == b'error'
    assert calls[0][1]['shell'] is False
    assert calls[0][1]['stdin'] == subprocess.DEVNULL
    for argv in (('powershell', '-Command', 'anything'), ('docker', 'system', 'prune'), ('docker', 'start', NAME)):
        with pytest.raises(ValueError):
            allowed.run(argv, timeout_seconds=1, max_output_bytes=100)


RESULT = '/tmp/llmbench-result.json'


def test_result_read_is_the_only_exec_shape_and_returns_raw_stdout():
    calls = []
    def factory(argv, **kwargs):
        calls.append((argv, kwargs))
        return FakeProcess(stdout=b'{"protocol":1}', stderr=b'')
    executor = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE, popen_factory=factory)
    result = executor.run(('docker', 'exec', NAME, 'cat', RESULT), timeout_seconds=1, max_output_bytes=100)
    assert result.status == 'completed' and result.returncode == 0
    assert result.stdout == b'{"protocol":1}' and result.stderr == b''
    assert calls[0][0] == ('docker', 'exec', NAME, 'cat', RESULT)
    assert calls[0][1]['shell'] is False and calls[0][1]['stdin'] == subprocess.DEVNULL


@pytest.mark.parametrize('argv', [
    ('docker', 'exec'),
    ('docker', 'exec', NAME),
    ('docker', 'exec', NAME, 'sh'),
    ('docker', 'exec', NAME, 'cat'),
    ('docker', 'exec', NAME, 'sh', '-c', 'cat ' + RESULT),
    ('docker', 'exec', NAME, 'sh', RESULT),
    ('docker', 'exec', NAME, 'cat', RESULT, '/etc/passwd'),
    ('docker', 'exec', NAME, 'cat', RESULT, RESULT),
    ('docker', 'exec', NAME, 'cat', '/tmp/llmbench-result.pending'),
    ('docker', 'exec', NAME, 'cat', '/tmp/../etc/passwd'),
    ('docker', 'exec', NAME, 'cat', '/workspace/.llmbench-input.json'),
    ('docker', 'exec', NAME, 'cat', RESULT + '\n'),
    ('docker', 'exec', NAME, 'cat', ' ' + RESULT),
    ('docker', 'exec', '-i', NAME, 'cat', RESULT),
    ('docker', 'exec', '--user=0', NAME, 'cat'),
    ('docker', 'exec', '--privileged', NAME, 'cat', RESULT),
    ('docker', 'exec', '-u', '0', NAME),
    ('docker', 'exec', 'other-container', 'cat', RESULT),
    ('docker', 'exec', 'llmbench-' + 'g' * 32, 'cat', RESULT),
    ('docker', 'exec', 'llmbench-' + '1' * 31, 'cat', RESULT),
    ('docker', 'exec', NAME.upper(), 'cat', RESULT),
    ('docker', 'exec', NAME + '\n', 'cat', RESULT),
    ('docker', 'exec', 'x' + NAME, 'cat', RESULT),
    ('docker', 'container', 'exec', NAME, 'cat', RESULT),
    # docker cp cannot read tmpfs and is no longer part of the vocabulary at all.
    ('docker', 'cp', NAME + ':' + RESULT, '-'),
    ('docker', 'cp', NAME + ':/etc/passwd', '-'),
])
def test_exec_and_cp_outside_the_exact_result_read_are_rejected_before_launch(argv):
    calls = []
    executor = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE,
                                      popen_factory=lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        executor.run(argv, timeout_seconds=1, max_output_bytes=100)
    assert calls == []


def test_oversized_result_read_is_killed_at_the_output_budget():
    process = FakeProcess(stdout=b'a' * 200_000, stderr=b'', running=True)
    executor = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE,
                                      popen_factory=lambda *args, **kwargs: process)
    result = executor.run(('docker', 'exec', NAME, 'cat', RESULT), timeout_seconds=1, max_output_bytes=1024)
    assert result.status == 'output-limit' and process.killed
    assert len(result.stdout) + len(result.stderr) == 1024


def test_output_flood_cannot_deadlock_or_exceed_budget():
    process = FakeProcess(stdout=b'a' * 200_000, stderr=b'b' * 200_000, running=True)
    executor = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE,
                                      popen_factory=lambda *args, **kwargs: process)
    result = executor.run(('docker', 'rm', '-f', NAME), timeout_seconds=1, max_output_bytes=1024)
    assert result.status == 'output-limit' and process.killed
    assert len(result.stdout) + len(result.stderr) == 1024


def test_timeout_kills_only_the_owned_client():
    process = FakeProcess(running=True)
    ticks = iter((0, 2))
    executor = BoundedProcessExecutor(session_lock=LOCK, mode=RunMode.LIVE,
                                      popen_factory=lambda *args, **kwargs: process, clock=lambda: next(ticks))
    result = executor.run(('docker', 'rm', '-f', NAME), timeout_seconds=1, max_output_bytes=1024)
    assert result.status == 'timeout' and process.killed


def test_real_executor_validates_worker_argv_without_launch(tmp_path):
    root = tmp_path / 'candidates'
    candidate = root / 'one'
    candidate.mkdir(parents=True)
    worker = DockerWorker(allowed_root=root, session_lock=LOCK)
    job = worker.prepare(candidate, image='local/test@sha256:' + 'a' * 64,
                         command=('python3', '/workspace/.llmbench-driver.py'), collect_result=True)
    BoundedProcessExecutor._validate(job.argv)
    assert not worker.synthetic
    changed = list(job.argv)
    changed.insert(6, '--network=host')
    with pytest.raises(ValueError):
        BoundedProcessExecutor._validate(tuple(changed))
