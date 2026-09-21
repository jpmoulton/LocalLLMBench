"""Trusted container entry scripts. These strings are never executed on the host.

Only case inputs and a source entrypoint are staged. Expected outputs remain in
the controller. User prints go to diagnostic stderr; scores use a separate file.
"""

PYTHON_DRIVER = r'''
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

input_path = Path('/workspace/.llmbench-input.json')
config = json.loads(input_path.read_text())
result_path = Path('/tmp/llmbench-result.json')
temporary_result_path = Path('/tmp/llmbench-result.pending')
write_text = temporary_result_path.write_text
publish_result = temporary_result_path.replace
sleep = time.sleep
dumps = json.dumps
original_args = json.loads(config['arguments_json'])
before = copy.deepcopy(original_args)
result = {'protocol': 1, 'case_id': config['case_id'], 'before': before,
          'compile_ok': False, 'runner_error': None}
node_finished = False
child_deadline = time.monotonic() + config['child_timeout_seconds']

def child_time_remaining():
    remaining = child_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Driver child budget exhausted before next process admission')
    return remaining

try:
    language = config['language']
    if language == 'python':
        # Avoid loading __pycache__ and redirect ordinary candidate prints away
        # from the result protocol. The dedicated result file is written last.
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location('llmbench_candidate', '/workspace/' + config['source'])
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
            function = getattr(module, config['function'])
            result['compile_ok'] = True
            print('LLMBENCH_CANDIDATE_ENTERED:' + config['case_id'], file=sys.stderr, flush=True)
            try:
                value = function(*original_args)
            except Exception as error:
                value = {'error': type(error).__name__}
        result['value'] = value
        result['after'] = original_args
    else:
        source = '/workspace/' + config['source']
        if language == 'typescript':
            os.mkdir('/tmp/compiled')
            completed = subprocess.run(
                ['tsc', '--strict', '--target', 'ES2020', '--module', 'commonjs',
                 '--noEmitOnError', '--rootDir', '/workspace', '--outDir', '/tmp/compiled', source],
                stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
                shell=False, timeout=child_time_remaining())
            if completed.returncode != 0:
                raise RuntimeError('TypeScript compilation failed')
            source = '/tmp/compiled/' + Path(config['source']).with_suffix('.js').name
        child = subprocess.run(
            ['node', '/workspace/.llmbench-node.cjs', source],
            stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
            shell=False, timeout=child_time_remaining())
        if child.returncode != 0 or not result_path.is_file():
            raise RuntimeError('Node candidate driver did not complete')
        # Node writes its own dedicated protocol file. Host validates every field.
        node_finished = True
except BaseException as error:
    result['runner_error'] = type(error).__name__ + ': ' + str(error)
    result['after'] = original_args
try:
    if not node_finished:
        write_text(dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        publish_result(result_path)
    # Keep tmpfs mounted while the controller copies the ready result. This is
    # bounded even if the controller disappears; normal cleanup removes us first.
    sleep(30)
except BaseException:
    # A missing/invalid result is a failure. Never print a substitute success.
    sys.exit(2)
'''.lstrip()


TEST_MODULE_DRIVER = r'''
import contextlib
import importlib
import json
from pathlib import Path
import signal
import sys
import time

# Bind everything the protocol needs BEFORE candidate code is imported, as PYTHON_DRIVER does.
config = json.loads(Path('/workspace/.llmbench-input.json').read_text())
result_path = Path('/tmp/llmbench-result.json')
temporary_result_path = Path('/tmp/llmbench-result.pending')
write_text = temporary_result_path.write_text
publish_result = temporary_result_path.replace
sleep = time.sleep
dumps = json.dumps
alarm = signal.alarm
result = {'protocol': 2, 'item_id': config['item_id'], 'compile_ok': False, 'runner_error': None,
          'error_origin': None, 'outcomes': []}


class _TestTimeout(BaseException):
    pass


def _expired(signum, frame):
    raise _TestTimeout()


signal.signal(signal.SIGALRM, _expired)
sys.dont_write_bytecode = True
sys.path.insert(0, '/workspace')
phase = 'candidate'
try:
    with contextlib.redirect_stdout(sys.stderr):
        alarm(config['import_timeout_seconds'])
        candidate = importlib.import_module('solution')
        if config.get('entry_point'):
            getattr(candidate, config['entry_point'])
        phase = 'harness'
        module = importlib.import_module(config['test_module'])
        alarm(0)
        result['compile_ok'] = True
        phase = 'candidate'
        for test_id, name in module.LLMBENCH_TESTS:
            outcome, detail = 'passed', None
            alarm(config['per_test_timeout_seconds'])
            try:
                getattr(module, name)()
            except _TestTimeout:
                outcome = 'timeout'
            except AssertionError as error:
                outcome, detail = 'assertion', str(error)[:200]
            except BaseException as error:
                outcome, detail = 'exception', (type(error).__name__ + ': ' + str(error))[:200]
            finally:
                alarm(0)
            result['outcomes'].append([test_id, outcome, detail])
            if outcome == 'timeout':  # pass@1 is already lost; do not spend the container budget on the rest
                break
except _TestTimeout:
    result['runner_error'] = 'TimeoutError: import exceeded the import budget'
    result['error_origin'] = phase
except BaseException as error:
    alarm(0)
    result['runner_error'] = type(error).__name__ + ': ' + str(error)[:500]
    result['error_origin'] = phase
try:
    write_text(dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    publish_result(result_path)
    sleep(30)  # keep tmpfs mounted while the controller copies the ready result
except BaseException:
    sys.exit(2)
'''.lstrip()


POLYGLOT_DRIVER = r'''
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# An Exercism exercise is run by its own test runner as a CHILD process, so candidate code never shares this
# interpreter. /workspace is mounted read-only; the runners want a writable tree, so it is copied to tmpfs.
config = json.loads(Path('/workspace/.llmbench-input.json').read_text())
result_path = Path('/tmp/llmbench-result.json')
pending_path = Path('/tmp/llmbench-result.pending')
result = {'protocol': 3, 'item_id': config['item_id'], 'exit_code': None, 'timed_out': False,
          'runner_error': None, 'output': ''}
try:
    # Verify the pinned runner without importing any candidate file. Missing tools are harness failures.
    preflight = subprocess.run(config['preflight_argv'], cwd='/tmp', stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False)
    if preflight.returncode != 0:
        raise RuntimeError('test toolchain unavailable: ' + preflight.stdout[-1000:].decode('utf-8', errors='replace'))
    work = '/tmp/work'
    shutil.copytree('/workspace', work, ignore=shutil.ignore_patterns('.llmbench-*'))
    for name, target in config['links'].items():
        os.symlink(target, os.path.join(work, name))
    environment = {'PATH': os.environ.get('PATH', ''), 'HOME': '/tmp', 'TMPDIR': '/tmp', 'CI': 'true',
                   'PYTHONDONTWRITEBYTECODE': '1', 'BABEL_DISABLE_CACHE': '1', 'NODE_ENV': 'test',
                   'NO_COLOR': '1', 'FORCE_COLOR': '0'}
    log_path = '/tmp/llmbench-test-output.txt'
    with open(log_path, 'wb') as log:
        child = subprocess.Popen(config['argv'], cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
        try:
            result['exit_code'] = child.wait(timeout=config['child_timeout_seconds'])
        except subprocess.TimeoutExpired:
            result['timed_out'] = True
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except OSError:
                pass
            child.wait(timeout=10)
    raw = Path(log_path).read_bytes()
    limit = config['max_output_chars']
    text = raw.decode('utf-8', errors='replace')
    result['output'] = text if len(text) <= limit else text[:limit // 4] + '\n...[truncated]...\n' + text[-(3 * limit) // 4:]
except BaseException as error:
    result['runner_error'] = type(error).__name__ + ': ' + str(error)[:500]
try:
    pending_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    pending_path.replace(result_path)
    time.sleep(30)  # keep tmpfs mounted while the controller copies the ready result
except BaseException:
    sys.exit(2)
'''.lstrip()


NODE_DRIVER = r'''
'use strict';
const fs = require('node:fs');
const stringify = JSON.stringify;
const parse = JSON.parse;
const writeFile = fs.writeFileSync.bind(fs);
const rename = fs.renameSync.bind(fs);
const config = parse(fs.readFileSync('/workspace/.llmbench-input.json', 'utf8'));
const args = parse(config.arguments_json);
const before = parse(stringify(args));
const result = {protocol: 1, case_id: config.case_id, before, compile_ok: false, runner_error: null};
try {
  const candidate = require(process.argv[2]);
  if (typeof candidate[config.function] !== 'function') throw new TypeError('Expected exported function');
  result.compile_ok = true;
  fs.writeSync(2, 'LLMBENCH_CANDIDATE_ENTERED:' + config.case_id + '\n');
  try { result.value = candidate[config.function](...args); }
  catch (error) { result.value = {error: error.name}; }
  result.after = args;
} catch (error) {
  result.runner_error = error.name + ': ' + error.message;
  result.after = args;
}
writeFile('/tmp/llmbench-result.pending', stringify(result), {encoding: 'utf8'});
rename('/tmp/llmbench-result.pending', '/tmp/llmbench-result.json');
'''.lstrip()
