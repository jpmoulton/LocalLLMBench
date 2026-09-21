"""Production private-fixture orchestration; all candidate execution stays in Docker."""

from __future__ import annotations

import base64
import math
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from ..config import canonical_json
from ..evaluations.tools import strict_json_loads
from .drivers import NODE_DRIVER, PYTHON_DRIVER
from .fixtures import CodingFixture, write_candidate
from .sandbox import DockerWorker, SandboxLimits


def stage_case(directory: Path, fixture: CodingFixture, check, patch: Mapping[str, str] | None,
               limits: SandboxLimits) -> Path:
    if fixture.language not in {'python', 'typescript', 'javascript'}:
        raise ValueError('Unsupported fixture language')
    module, separator, function = fixture.entrypoint.partition(':')
    if (not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', module)
            or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', function)):
        raise ValueError('Fixture entrypoint must be an explicit module:function')
    extension = {'python': '.py', 'typescript': '.ts', 'javascript': '.cjs'}[fixture.language]
    source = module + extension
    if source not in {item.path for item in fixture.initial_files}:
        raise ValueError('Entrypoint source is absent from the fixture')
    if any(item.path.startswith('.llmbench-') for item in fixture.initial_files):
        raise ValueError('Fixture cannot use reserved driver filenames')
    arguments = strict_json_loads(check.arguments_json)
    if type(arguments) is not list:
        raise ValueError('Fixture arguments must be a JSON argument array')
    path = write_candidate(directory, fixture, patch)
    path.joinpath('.llmbench-input.json').write_text(canonical_json({
        'case_id': check.case_id, 'language': fixture.language, 'source': source, 'function': function,
        'arguments_json': check.arguments_json, 'child_timeout_seconds': max(1, limits.timeout_seconds - 2),
    }), encoding='utf-8')
    path.joinpath('.llmbench-driver.py').write_text(PYTHON_DRIVER, encoding='utf-8')
    path.joinpath('.llmbench-node.cjs').write_text(NODE_DRIVER, encoding='utf-8')
    return path


def validate_observation(payload: bytes | None, *, check) -> dict:
    if payload is None:
        return {'passed': False, 'status': 'result-missing'}
    try:
        parsed = strict_json_loads(payload.decode('utf-8'))
        if type(parsed) is not dict or set(parsed) - {
            'protocol', 'case_id', 'before', 'after', 'value', 'compile_ok', 'runner_error'
        }:
            raise ValueError('Unexpected result protocol fields')
        if type(parsed.get('protocol')) is not int or parsed['protocol'] != 1 or parsed.get('case_id') != check.case_id:
            raise ValueError('Result protocol/case identity mismatch')
        if type(parsed.get('compile_ok')) is not bool:
            raise ValueError('Missing actual compilation evidence')
        original = strict_json_loads(check.arguments_json)
        intact = ('before' in parsed and 'after' in parsed
                  and canonical_json(parsed['before']) == canonical_json(original)
                  and canonical_json(parsed['after']) == canonical_json(original))
        value_matches = ('value' in parsed and canonical_json(parsed['value'])
                         == canonical_json(strict_json_loads(check.expected_json)))
        passed = (parsed['compile_ok'] and parsed.get('runner_error') is None and intact and value_matches)
        status = 'passed' if passed else 'incorrect'
        if isinstance(parsed.get('runner_error'), str) and parsed['runner_error'].startswith('FileNotFoundError:'):
            status = 'environment-error'
        return {'passed': passed, 'status': status,
                'input_unchanged': intact, 'value_matches': value_matches,
                'compile_ok': parsed['compile_ok'], 'observation': parsed}
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        return {'passed': False, 'status': 'protocol-error', 'error': str(exc)}


def run_coding_fixture(fixture: CodingFixture, patch: Mapping[str, str] | None, *,
                       worker: DockerWorker, image: str, limits: SandboxLimits | None = None,
                       split: str = 'development', first_attempt: bool = True,
                       timeout_seconds: float | None = None) -> dict:
    """Execute each check in a fresh bounded container and retain every required case.

    Returns a controller-compatible sample plus exact JSON trace bytes for an
    artifact. No scoring claims are drawn from stdout/stderr or exit code alone.
    Images must already contain Python 3; JS/TS also need Node and TS needs tsc.
    """
    worker.session_lock.check('container', worker.mode)
    if split not in {'development', 'holdout'} or type(first_attempt) is not bool:
        raise ValueError('Invalid split or first-attempt flag')
    if not fixture.checks or len({case.case_id for case in fixture.checks}) != len(fixture.checks):
        raise ValueError('Fixture checks must have distinct nonempty IDs')
    resources = limits or SandboxLimits()
    if timeout_seconds is not None and (type(timeout_seconds) not in (float, int)
                                        or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('Total fixture timeout must be finite and positive')
    root = worker.allowed_root / ('fixture-' + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    started, cases, traces = time.monotonic(), [], []
    deadline = started + timeout_seconds if timeout_seconds is not None else None
    halted = None
    for index, check in enumerate(fixture.checks):
        case_started = time.monotonic()
        remaining = deadline - case_started if deadline is not None else None
        if remaining is not None and remaining < 1:
            cases.append({'case_id': check.case_id, 'passed': False,
                          'status': 'fixture-time-budget', 'elapsed_seconds': 0.0})
            continue
        if halted:
            cases.append({'case_id': check.case_id, 'passed': False,
                          'status': 'not-run-after-cleanup-failure' if halted == 'cleanup-unverified'
                          else 'not-run-after-environment-error', 'elapsed_seconds': 0.0})
            continue
        job = None
        try:
            current_limits = resources
            if remaining is not None:
                current_limits = SandboxLimits.model_validate({**resources.model_dump(),
                                                               'timeout_seconds': min(resources.timeout_seconds, int(remaining))})
            path = stage_case(root / f'case-{index:03d}', fixture, check, patch, current_limits)
            job = worker.prepare(path, image=image, command=('python3', '/workspace/.llmbench-driver.py'),
                                 limits=current_limits, collect_result=True)
            result = worker.run(job)
            raw = asdict(result)
            for key in ('stdout', 'stderr', 'result_bytes', 'result_read_stdout', 'result_read_stderr'):
                raw[key + '_base64'] = base64.b64encode(raw.pop(key) or b'').decode('ascii')
            traces.append({'case_id': check.case_id, 'job_argv': job.argv, 'worker': raw})
            if result.status != 'completed' or result.returncode != 0:
                outcome = {'passed': False, 'status': result.status if result.status != 'completed' else 'execution-error'}
            else:
                outcome = validate_observation(result.result_bytes, check=check)
            outcome['cleanup_confirmed'] = result.cleanup_confirmed
            if not result.cleanup_confirmed:
                outcome.update(passed=False, status='cleanup-unverified', cleanup_error=result.cleanup_error)
                halted = 'cleanup-unverified'
        except Exception as exc:
            outcome = {'passed': False, 'status': 'cleanup-unverified' if job is not None else 'environment-error',
                       'error': str(exc)}
            traces.append({'case_id': check.case_id, 'error': type(exc).__name__ + ': ' + str(exc)})
            halted = outcome['status']
        outcome.update(case_id=check.case_id, elapsed_seconds=time.monotonic() - case_started)
        cases.append(outcome)
    passed = sum(case['passed'] for case in cases)
    sample = {'task_id': fixture.fixture_id, 'fixture_hash': fixture.identity(), 'category': 'coding',
              'suite_revision': fixture.revision, 'split': split, 'language': fixture.language,
              'status': 'completed', 'score': float(passed == len(cases)), 'passed': passed == len(cases),
              'required_checks': len(fixture.checks), 'attempted_checks': len(traces), 'passed_checks': passed,
              'success_all_required': passed / len(fixture.checks), 'cases': cases,
              'first_attempt_success': first_attempt and passed == len(cases),
              'elapsed_seconds': time.monotonic() - started, 'synthetic': worker.synthetic,
              'execution': 'isolated-docker-per-case'}
    abort_campaign = any(case.get('status') in {'cleanup-unverified', 'not-run-after-cleanup-failure'} for case in cases)
    sample['abort_campaign'] = abort_campaign
    return {'sample': sample, 'raw_bytes': canonical_json({'fixture': fixture.identity(), 'traces': traces}).encode(),
            'abort_campaign': abort_campaign,
            'candidate_path': str(root)}
