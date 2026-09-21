"""Root-owned live coding-worker calibration; no model and no host candidate execution.

Every completed variant is durably recorded immediately. Acceptance requires working
references, actual wrong-output negatives, actual candidate timeouts, and verified
absence of every container owned by this invocation. Infrastructure failures fail it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from llmbench.coding.acceptance import acceptance_summary, acceptance_verdict
from llmbench.coding.fixtures import fixtures
from llmbench.coding.image_plan import verified_local_image_id
from llmbench.coding.runner import run_coding_fixture
from llmbench.coding.sandbox import DockerWorker, SandboxLimits
from llmbench.config import RunMode
from llmbench.safety import SessionLock

HANGS = {"python": "def chunked(values, size):\n    while True:\n        pass\n",
         "typescript": "export type RecordItem = { key: string; value: number };\nexport function groupRecords("
                       "records: RecordItem[]): {key:string; values:number[]}[] { for (;;) {} }\n",
         "javascript": "exports.stableUnique = values => { for (;;) {} };\n"}


def durable_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    temporary = path.with_name(path.name + '.pending') if replace else path
    with temporary.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iidfile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hang-timeout', type=int, default=12)
    args = parser.parse_args()
    hang_limits = SandboxLimits(timeout_seconds=args.hang_timeout)
    image = verified_local_image_id(args.iidfile)
    lock = SessionLock.read('runtime-policy.json')
    lock.check('container', RunMode.LIVE)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    worker = DockerWorker(allowed_root=out / 'work', session_lock=lock, mode=RunMode.LIVE)
    (out / 'work').mkdir()
    rows, names, absence_checks = [], set(), []
    selected = fixtures()
    expected_rows = 3 * len(selected)
    error = None

    def checkpoint() -> dict:
        summary = acceptance_summary(rows, expected_rows=expected_rows, absence_checks=absence_checks,
                                     image=image, error=error)
        durable_write(out / 'acceptance-summary.json', json.dumps(summary, indent=2).encode(), replace=True)
        return summary

    checkpoint()
    try:
        halted = False
        for fixture in selected:
            reference = {item.path: item.content for item in fixture.reference_files}
            hang = {fixture.initial_files[0].path: HANGS[fixture.language]}
            for label, patch, limits in (('reference', reference, None), ('initial', None, None),
                                         ('hang', hang, hang_limits)):
                started = time.monotonic()
                result = run_coding_fixture(fixture, patch, worker=worker, image=image, limits=limits)
                sample = result['sample']
                for trace in json.loads(result['raw_bytes']).get('traces', []):
                    argv = trace.get('job_argv', [])
                    if '--name' in argv:
                        names.add(argv[argv.index('--name') + 1])
                name = f"{fixture.fixture_id.replace('/', '-')}-{label}"
                durable_write(out / f'{name}.trace.json', result['raw_bytes'])
                row = {'fixture': fixture.fixture_id, 'variant': label,
                       'expected_pass': label == 'reference', 'passed': sample['passed'],
                       **acceptance_verdict(fixture, label, result),
                       'synthetic': sample['synthetic'], 'abort_campaign': result['abort_campaign'],
                       'cases': sample['cases'], 'elapsed_seconds': round(time.monotonic() - started, 2)}
                rows.append(row)
                durable_write(out / f'{name}.result.json', json.dumps(row, indent=2).encode())
                checkpoint()
                print(json.dumps(row), flush=True)
                # Without a working reference the negative controls are not useful;
                # stop immediately on cleanup uncertainty or failed calibration.
                if result['abort_campaign'] or not row['as_expected']:
                    halted = True
                    break
            if halted:
                break
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        for name in sorted(names):
            try:
                check = worker.executor.run(('docker', 'ps', '--all', '--filter', f'name=^/{name}$',
                                             '--format', '{{.ID}}'), timeout_seconds=10, max_output_bytes=4096)
                queried = check.status == 'completed' and check.returncode == 0
                absence_checks.append({'name': name, 'status': check.status, 'returncode': check.returncode,
                                       'verified_absent': queried and not check.stdout.strip(),
                                       'present': bool(check.stdout.strip()) if queried else None,
                                       'stderr': check.stderr.decode('utf-8', errors='replace')})
            except BaseException as exc:
                absence_checks.append({'name': name, 'verified_absent': False,
                                       'error': type(exc).__name__ + ': ' + str(exc)})
        summary = checkpoint()
    print('ACCEPTED:', summary['accepted'], '| complete:', summary['complete'],
          '| cleanup:', summary['all_cleanup_confirmed'], '| absence:', summary['absence_query_verified'], flush=True)
    raise SystemExit(0 if summary['accepted'] else 1)


if __name__ == '__main__':
    main()
