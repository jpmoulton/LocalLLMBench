"""The ``llmbench`` command line.

``validate``, ``capabilities`` and ``plan`` are pure; ``candidate``, ``prepare``, ``tune`` and ``resume`` start
containers, and only when ``runtime-policy.json`` allows it. ``doctor``, ``list``, ``show``, ``analyze`` and
``report`` read recorded evidence and never touch a server, a GPU or Docker.

Exit codes: 0 ok, 2 invalid input or policy, 3 run did not complete, 4 cleanup uncertain (stop the campaign).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from ..backends.base import UnsupportedSetting
from ..safety import OperationForbidden
from .capabilities import ALLOWED_FLAGS, check_argv, load_capabilities, require_supported
from .config import ImageBundle, ImageRef, read_image_bundle, read_run_config
from .plan import build_compose_plan, build_server_argv

PROG = "llmbench"
DEFAULT_CAPABILITIES = "artifacts/container-prep"
PREVIEW_ATTEMPT = "0" * 32
PREVIEW_GRANT = {"grant_seconds": 1, "artifact_bytes": 0, "issued_utc": "preview", "issued_host_offset_seconds": 0.0}
SELECTION_PLAN_NAME = "benchmark-selection.json"
"""Written into a ``tune --model`` output directory: which public benchmarks were offered, selected and skipped,
with the reason for each. The selections themselves are in ``session-config.json``; this is why anything is
absent."""


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog=PROG, description="Tune and benchmark llama.cpp serving settings in "
                                   "containers. Live commands require runtime-policy.json; optimize and sample "
                                   "also support --plan-only.")
    sub = root.add_subparsers(dest="command", required=True)
    for name, text in (("validate", "Validate a candidate configuration; no I/O beyond reading it"),
                       ("capabilities", "Parse the saved llama-server help/version; optionally check a configuration"),
                       ("plan", "Print the exact server argv and Compose project without running anything"),
                       ("candidate", "Run one bounded candidate and write evidence and reports")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--config", required=name != "capabilities")
        if name != "validate":
            command.add_argument("--capabilities-dir", default=DEFAULT_CAPABILITIES)
    candidate = sub.choices["candidate"]
    candidate.add_argument("--output", required=True)
    candidate.add_argument("--policy", default="runtime-policy.json")
    candidate.add_argument("--budget-seconds", type=float)
    candidate.add_argument("--image-bundle", help="image-bundle.json from `prepare`; config image refs must match")
    prepare = sub.add_parser("prepare", help="Pull the inference image by digest, build and self-check the evaluator "
                                             "image, record image-bundle.json (requires container policy)")
    prepare.add_argument("--inference", required=True, help="repository@sha256:<digest> of the llama.cpp server image")
    prepare.add_argument("--evaluator-base", required=True, help="python:3.12-slim-bookworm@sha256:<digest>")
    prepare.add_argument("--worker-iidfile", help="worker-image.id from the coding worker build")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--policy", default="runtime-policy.json")
    prepare.add_argument("--wheel", help="prebuilt localllmbench wheel; default builds one from this tree")
    prepare.add_argument("--lock", help="hash-pinned Linux lock; default is the packaged requirements.linux.lock")
    from ..evidence_cli import add_evidence_commands
    from .session import add_session_commands
    from .optimization import add_optimization_commands
    from .sampling import add_sampling_commands
    add_session_commands(sub)
    add_evidence_commands(sub)
    add_optimization_commands(sub)
    add_sampling_commands(sub)
    _add_dataset_root(sub.choices["tune"])
    return root


def _add_dataset_root(tune: argparse.ArgumentParser) -> None:
    """``tune --dataset-root``: where the pinned public-benchmark corpora live for this session.

    Added beside the session parser rather than forking it (``containers/session.py`` owns tune/resume), and
    guarded so it stays a no-op if that parser ever declares the flag itself.
    """
    from .derive import STAGED_DATASET_DIRNAME
    try:
        tune.add_argument("--dataset-root", default=None, metavar="DIR",
                          help="directory holding the staged public-benchmark corpora (bfcl/, ruler/, evalplus/, "
                               f"aider-polyglot/). Default: {STAGED_DATASET_DIRNAME} when it exists and the "
                               "evaluator is a host process, otherwise the evaluator image's baked path. "
                               "--model only; a directory that does not exist is refused, never guessed.")
    except argparse.ArgumentError:  # the session parser already declares it
        pass


def _default_runner(args):
    from .runner import ContainerRunner
    return ContainerRunner(capabilities_dir=args.capabilities_dir, policy_path=args.policy)


def _default_preparer(args) -> ImageBundle:
    """Docker runs only through a preparation-enabled executor authorized by the persistent policy."""
    from ..config import RunMode
    from .executor import ComposeExecutor
    from .image_plan import PrepareInputs, build_wheel, prepare_images
    from ..safety import SessionLock
    lock = SessionLock.read(args.policy)
    lock.check("container", RunMode.LIVE)
    executor = ComposeExecutor(session_lock=lock, mode=RunMode.LIVE, allow_preparation=True)
    output = Path(args.output)
    wheel = Path(args.wheel) if args.wheel else build_wheel(Path(__file__).resolve().parents[3], output / "wheel")
    inputs = PrepareInputs(inference=args.inference, evaluator_base=args.evaluator_base, wheel_path=wheel,
                           lock_path=args.lock, worker_iidfile=args.worker_iidfile)
    return prepare_images(inputs, executor=executor, artifact_dir=output, session_lock=lock)


def _same_image(configured: ImageRef | None, bundled: ImageRef | None, role: str) -> list[str]:
    if configured is None:
        return []
    if bundled is None:
        return [f"{role}: the bundle carries no {role} image"]
    problems = []
    for field in ("reference", "image_id", "entrypoint", "help_sha256", "build_info"):
        if getattr(configured, field) != getattr(bundled, field):
            problems.append(f"{role}.{field}: config {getattr(configured, field)!r} != bundle "
                            f"{getattr(bundled, field)!r}")
    return problems


def check_image_bundle(config, bundle: ImageBundle) -> list[str]:
    """Every image the config names must be exactly the prepared one; extra bundle images are fine."""
    problems = _same_image(config.inference_image, bundle.inference, "inference")
    problems += _same_image(config.evaluator.image, bundle.evaluator, "evaluator")
    problems += _same_image(config.worker_image, bundle.worker, "worker")
    # Host evaluation uses the installed registry, checked by host preflight; the bundle records
    # the evaluator image registry and is authoritative only when that image actually evaluates.
    if (config.evaluator.mode == "container" and config.registry_digest is not None
            and config.registry_digest != bundle.registry_digest):
        problems.append("registry_digest: config differs from the bundle's evaluator registry digest")
    return problems


def _record_plan(output: str | Path, plan: dict) -> None:
    """Best-effort durable copy of the benchmark plan inside a started session directory.

    Written only where the session already exists, so a policy refusal still leaves nothing behind (the
    console line printed before the run is the up-front record). A failure here never changes the exit code:
    the plan is evidence about the session, not part of running it.
    """
    try:
        target = Path(output)
        if target.is_dir():
            target.joinpath(SELECTION_PLAN_NAME).write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
    except OSError as exc:
        print(f"{PROG} tune: the benchmark plan could not be recorded: {exc}", file=sys.stderr)


def _run_tune(args, main_tune) -> int:
    """Show the benchmark plan, apply ``--dataset-root``, then hand over to the session command.

    ``session.main_tune`` derives the session itself from ``--base-config`` (which it reads from disk) and charges
    the derivation and the GGUF hashing to the session wall. Rather than fork that, ``--dataset-root`` is applied
    by handing it the SAME base config with ``dataset_root`` set: one field, in the file the session already
    reads, whose value then round-trips through ``session-config.json`` into every candidate and into ``resume``.

    The plan is computed here from the GGUF headers only (no hashing) so it can be shown BEFORE a four-hour run
    starts; the cost is one extra header read per GGUF, outside the session budget. It comes from the same
    functions ``derive_session_config`` uses with the same arguments, so it cannot describe a different session.
    """
    from .derive import benchmark_plan_for_model, benchmark_plan_lines
    if args.config:
        if getattr(args, "dataset_root", None) is not None:
            raise ValueError("--dataset-root derives a session from --model; a --config session already states "
                             "its own base.dataset_root")
        return main_tune(args)
    explicit = getattr(args, "dataset_root", None)
    base = read_run_config(args.base_config)
    plan = benchmark_plan_for_model(args.model, base=base, budget_seconds=args.budget_seconds or 14400,
                                    context_floor=getattr(args, "context_floor", None),
                                    context_ceiling=getattr(args, "context_ceiling", None),
                                    dataset_root=explicit)
    for line in benchmark_plan_lines(plan):
        print(f"{PROG} tune: {line}", file=sys.stderr)
    if plan["dataset_root_source"] != "explicit":  # derivation resolves the same default itself
        code = main_tune(args)
    else:
        raw = {**json.loads(Path(args.base_config).read_text(encoding="utf-8")),
               "dataset_root": plan["dataset_root"]}
        with tempfile.TemporaryDirectory(prefix="llmbench-base-") as directory:
            overlaid = Path(directory) / "base-config.json"
            overlaid.write_text(json.dumps(raw), encoding="utf-8")
            args.base_config = str(overlaid)
            code = main_tune(args)
    _record_plan(args.output, plan)
    return code


def main(argv=None, *, runner_factory=None, preparer=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "optimize":
        from .optimization import main_optimize
        return main_optimize(args)
    if args.command == "sample":
        from .sampling import main_sampling
        return main_sampling(args)
    if args.command in ("tune", "resume"):  # session commands take a ContainerSessionConfig, not a run config
        from .session import main_resume, main_tune
        if args.command == "resume":
            return main_resume(args)
        try:
            return _run_tune(args, main_tune)
        except (ValidationError, ValueError, KeyError, OSError, OperationForbidden) as exc:
            print(f"{PROG} tune: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    try:
        from ..evidence_cli import COMMANDS as EVIDENCE_COMMANDS, run_evidence_command
        if args.command in EVIDENCE_COMMANDS:
            print(json.dumps(run_evidence_command(args), indent=2, ensure_ascii=False))
            return 0
        config = read_run_config(args.config) if getattr(args, "config", None) else None
        if args.command == "validate":
            data = {"valid": True, "label": config.label, "fingerprint": config.fingerprint(),
                    "alias": config.alias(), "execution_performed": False}
        elif args.command == "capabilities":
            caps = load_capabilities(args.capabilities_dir)
            data = {"flags": len(caps.flags), "help_sha256": caps.help_sha256, "version": caps.version,
                    "build": caps.build, "allowed_flags_missing": sorted(ALLOWED_FLAGS - set(caps.flags)),
                    "execution_performed": False}
            if config is not None:
                data["help_sha256_matches"] = caps.help_sha256 == config.inference_image.help_sha256
                data["findings"] = check_argv(build_server_argv(config), caps)
            if data["allowed_flags_missing"] or data.get("findings") or data.get("help_sha256_matches") is False:
                print(json.dumps(data, indent=2))
                return 2
        elif args.command == "plan":
            caps = load_capabilities(args.capabilities_dir)
            grant = None
            if config.evaluator.mode == "container":
                from .config import ChildGrant
                grant = ChildGrant(**{**PREVIEW_GRANT, "artifact_bytes": config.limits.evaluator_artifact_bytes,
                                      "watchdog_slack_seconds": config.evaluator.watchdog_slack_seconds})
            plan = build_compose_plan(config, "RUN_DIR", attempt_id=PREVIEW_ATTEMPT,
                                      session_id=config.session_id or "preview", grant=grant)
            require_supported(plan.server_argv, caps, expected_help_sha256=config.inference_image.help_sha256)
            data = {"execution_performed": False, "preview": "attempt id, RUN_DIR and the evaluator grant are "
                                                             "placeholders",
                    "fingerprint": config.fingerprint(), "server_argv": list(plan.server_argv),
                    "compose": plan.compose}
        elif args.command == "prepare":
            bundle = (preparer or _default_preparer)(args)
            data = {"execution_performed": True, "output": str(args.output),
                    "inference": bundle.inference.image_id, "evaluator": bundle.evaluator.image_id,
                    "worker": bundle.worker.image_id if bundle.worker else None,
                    "help_sha256": bundle.help_sha256, "registry_digest": bundle.registry_digest,
                    "prepared_utc": bundle.prepared_utc}
        else:
            if args.image_bundle:
                problems = check_image_bundle(config, read_image_bundle(args.image_bundle))
                if problems:
                    raise ValueError("config images differ from the prepared bundle: " + "; ".join(problems))
            runner = (runner_factory or _default_runner)(args)
            result = runner.run(config, args.output, remaining_budget_seconds=args.budget_seconds)
            data = {"state": result.state, "synthetic": result.synthetic, "attempt_id": result.attempt_id,
                    "failure_stage": result.failure_stage, "failure_reasons": list(result.failure_reasons),
                    "warnings": list(result.warnings), "cleanup_verified": result.cleanup.verified,
                    "abort_campaign": result.abort_campaign, "elapsed_seconds": result.elapsed_seconds,
                    "effective_settings_verified": result.effective_settings_verified,
                    "actual_context_verified": result.actual_context_verified,
                    "minimum_native_tps": result.speed.get("minimum_native_tps"),
                    "samples_total": result.samples_total, "reports": result.model_dump(mode="json")["reports"],
                    "output": str(args.output)}
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0 if result.state == "completed" else 4 if result.state == "cleanup-uncertain" else 3
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    except (ValidationError, ValueError, KeyError, OSError, OperationForbidden, UnsupportedSetting) as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
