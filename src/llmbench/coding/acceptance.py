"""Strict calibration oracle for real worker acceptance, never a model score."""

from __future__ import annotations

import base64
import json

from .fixtures import CodingFixture


def acceptance_verdict(fixture: CodingFixture, variant: str, result: dict) -> dict:
    if variant not in {'reference', 'initial', 'hang'}:
        raise ValueError('Unknown acceptance variant')
    sample = result['sample']
    reasons = []
    cases = sample.get('cases', [])
    ids = [case.get('case_id') for case in cases]
    expected_ids = [check.case_id for check in fixture.checks]
    if (ids != expected_ids or sample.get('required_checks') != len(expected_ids)
            or sample.get('attempted_checks') != len(expected_ids)):
        reasons.append('Every required case must actually run in fixture order')
    if sample.get('synthetic') is not False:
        reasons.append('Only actual Docker execution can pass live acceptance')
    if result.get('abort_campaign') is not False or sample.get('abort_campaign') is not False:
        reasons.append('Campaign abort or missing abort evidence')
    if not cases or any(case.get('cleanup_confirmed') is not True for case in cases):
        reasons.append('Every case requires explicit verified cleanup')
    if variant == 'reference':
        if (sample.get('passed') is not True or not cases
                or any(case.get('status') != 'passed' or case.get('passed') is not True
                       or case.get('compile_ok') is not True for case in cases)):
            reasons.append('Reference must compile and pass every actual check')
    elif variant == 'initial':
        if (sample.get('passed') is not False or not cases
                or not any(case.get('status') == 'incorrect' and case.get('passed') is False
                           and case.get('value_matches') is False for case in cases)
                or any(case.get('status') not in {'passed', 'incorrect'}
                       or case.get('compile_ok') is not True or case.get('input_unchanged') is not True
                       or not {'before', 'after', 'value'} <= case.get('observation', {}).keys()
                       or case.get('observation', {}).get('runner_error') is not None for case in cases)):
            reasons.append('Initial source must execute normally and fail an expected-output check')
    else:
        traces = json.loads(result['raw_bytes']).get('traces', [])
        by_id = {trace.get('case_id'): trace.get('worker', {}) for trace in traces}
        if sample.get('passed') is not False:
            reasons.append('Hanging source must fail')
        for case in cases:
            raw = by_id.get(case.get('case_id'), {})
            # A Docker startup/read timeout is infrastructure failure. Python's
            # known infinite loop qualifies only after completed missing-file
            # reads in a live container, followed by the publication deadline.
            try:
                missing = base64.b64decode(raw.get('result_read_stderr_base64', ''), validate=True)
            except ValueError:
                missing = b''
            publication_timeout = (
                fixture.language == 'python' and case.get('status') == 'timeout'
                and raw.get('status') == 'timeout' and raw.get('result_read_attempts', 0) > 0
                and raw.get('result_read_status') == 'completed' and raw.get('result_read_returncode') == 1
                and missing.strip() == b'cat: /tmp/llmbench-result.json: No such file or directory'
            )
            # JS/TS execute in a bounded child. A compiler timeout does not
            # prove that the hanging candidate ran; require the Node call.
            error = case.get('observation', {}).get('runner_error')
            child_timeout = (
                fixture.language in {'javascript', 'typescript'} and case.get('status') == 'incorrect'
                and isinstance(error, str) and error.startswith('TimeoutExpired:')
                and "['node', '/workspace/.llmbench-node.cjs'," in error
                and raw.get('status') == 'completed' and raw.get('returncode') == 0
            )
            try:
                stderr = base64.b64decode(raw.get('stderr_base64', ''), validate=True)
            except ValueError:
                stderr = b''
            entered = ('LLMBENCH_CANDIDATE_ENTERED:' + case['case_id']).encode() in stderr.splitlines()
            if case.get('passed') is not False or not entered or not (publication_timeout or child_timeout):
                reasons.append(f"{case.get('case_id')}: no verified candidate timeout")
    return {'as_expected': not reasons, 'reasons': reasons}


def acceptance_summary(rows: list[dict], *, expected_rows: int, absence_checks: list[dict],
                       image: str, error: str | None = None) -> dict:
    cleanup = bool(rows) and all(row['cases'] and all(case.get('cleanup_confirmed') is True
                        for case in row['cases']) for row in rows)
    absent = bool(absence_checks) and all(check.get('verified_absent') is True for check in absence_checks)
    complete = len(rows) == expected_rows
    expected = complete and all(row.get('as_expected') is True for row in rows)
    abort = any(row.get('abort_campaign') is not False for row in rows)
    return {'image': image, 'accepted': expected and cleanup and absent and not abort and error is None,
            'complete': complete, 'expected_rows': expected_rows, 'all_as_expected': expected,
            'all_cleanup_confirmed': cleanup, 'absence_query_verified': absent,
            'any_abort': abort, 'error': error, 'absence_checks': absence_checks,
            'leftover_worker_containers': [check['name'] for check in absence_checks
                                           if check.get('present') is True], 'rows': rows}
