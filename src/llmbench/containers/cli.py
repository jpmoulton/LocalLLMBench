"""The ``llmbench`` command line.

``validate``, ``capabilities`` and ``plan`` are pure; ``candidate``, ``prepare``, ``tune`` and ``resume`` start
containers -- or, for the ``metal-native`` runtime, a pinned llama-server on this Mac -- and only when
``runtime-policy.json`` allows it. ``doctor``, ``list``, ``show``, ``analyze`` and ``report`` read recorded
evidence and never touch a server, a GPU or Docker.

A run config names its own runtime (``ContainerRunConfig.runtime``, default ``nvidia-container``), so ``--runtime``
on a command that reads a config is an assertion, refused when it disagrees; it CHOOSES the runtime only where no
config exists yet (``capabilities`` alone, ``prepare``, ``tune --model``). With ``--runtime`` omitted every NVIDIA
default, output key and exit code is what it was before runtimes existed; the native keys appear only for a
metal-native config.

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
from .config import (DEFAULT_RUNTIME, RUNTIMES, ImageBundle, ImageRef, NativeBundle, read_image_bundle,
                     read_native_bundle, read_run_config)
from .plan import build_compose_plan, build_server_argv
from .runtime import NATIVE_RUNTIME, not_enforced_settings, required_operations, runtime_spec, unsupported_settings

PROG = "llmbench"
DEFAULT_CAPABILITIES = "artifacts/container-prep"
NATIVE_HOST = "127.0.0.1"
NATIVE_PORT_PLACEHOLDER = 0
"""A native server listens on loopback only, on a free port the runner picks when it starts the server; the pure
commands show this placeholder in its place (``--port`` takes any value, so the capability check is unaffected)."""
PREVIEW_ATTEMPT = "0" * 32
PREVIEW_GRANT = {"grant_seconds": 1, "artifact_bytes": 0, "issued_utc": "preview", "issued_host_offset_seconds": 0.0}
SELECTION_PLAN_NAME = "benchmark-selection.json"
"""Written into a ``tune --model`` output directory: which public benchmarks were offered, selected and skipped,
with the reason for each. The selections themselves are in ``session-config.json``; this is why anything is
absent."""


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog=PROG, description="Tune and benchmark llama.cpp serving settings in "
                                   "containers, or natively with Metal on Apple Silicon (--runtime metal-native). "
                                   "Live commands require runtime-policy.json; optimize and sample also support "
                                   "--plan-only.")
    sub = root.add_subparsers(dest="command", required=True)
    for name, text in (("validate", "Validate a candidate configuration; no I/O beyond reading it"),
                       ("capabilities", "Parse the saved llama-server help/version; optionally check a configuration"),
                       ("plan", "Print the exact server argv and Compose project without running anything"),
                       ("candidate", "Run one bounded candidate and write evidence and reports")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--config", required=name != "capabilities")
        command.add_argument("--runtime", choices=RUNTIMES, default=None,
                             help="the runtime the config must name (it states its own; a different one is "
                                  "refused)" if name != "capabilities" else
                                  "which runtime's prepared help to read when --capabilities-dir is not given "
                                  f"(default {DEFAULT_RUNTIME}); with --config it must agree with the config")
        if name != "validate":
            command.add_argument("--capabilities-dir", default=None,
                                 help=f"saved llama-server help/version. Default: {DEFAULT_CAPABILITIES} for "
                                      f"nvidia-container, {runtime_spec(NATIVE_RUNTIME).default_capabilities_dir} "
                                      "for metal-native" + (" (a native candidate reads the executable's own "
                                                            "--help at admission and takes no directory)"
                                                            if name == "candidate" else ""))
    candidate = sub.choices["candidate"]
    candidate.add_argument("--output", required=True)
    candidate.add_argument("--policy", default="runtime-policy.json")
    candidate.add_argument("--budget-seconds", type=float)
    candidate.add_argument("--image-bundle", help="image-bundle.json from `prepare`; config image refs must match")
    candidate.add_argument("--native-bundle", help="native-bundle.json from `prepare --runtime metal-native`; the "
                                                   "config's native server pins must match (metal-native only)")
    prepare = sub.add_parser("prepare", help="Pull the inference image by digest, build and self-check the evaluator "
                                             "image, record image-bundle.json (requires container policy); or, "
                                             "with --runtime metal-native, pin a local llama-server and record "
                                             "native-bundle.json (requires native policy)")
    prepare.add_argument("--runtime", choices=RUNTIMES, default=None,
                         help=f"what to prepare (default {DEFAULT_RUNTIME})")
    prepare.add_argument("--inference", help="repository@sha256:<digest> of the llama.cpp server image "
                                             "(nvidia-container; required)")
    prepare.add_argument("--evaluator-base", help="python:3.12-slim-bookworm@sha256:<digest> (nvidia-container; "
                                                  "required)")
    prepare.add_argument("--llama-server", metavar="PATH",
                         help="the llama-server executable to pin, e.g. from scripts/install_llamacpp_macos.py "
                              "(metal-native; required)")
    prepare.add_argument("--worker-iidfile", help="worker-image.id from the coding worker build")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--policy", default="runtime-policy.json")
    prepare.add_argument("--wheel", help="prebuilt localllmbench wheel; default builds one from this tree "
                                         "(nvidia-container)")
    prepare.add_argument("--lock", help="hash-pinned Linux lock; default is the packaged requirements.linux.lock "
                                        "(nvidia-container)")
    from ..evidence_cli import add_evidence_commands
    from .session import add_session_commands
    from .optimization import add_optimization_commands
    from .sampling import add_sampling_commands
    add_session_commands(sub)
    add_evidence_commands(sub)
    add_optimization_commands(sub)
    add_sampling_commands(sub)
    _add_dataset_root(sub.choices["tune"])
    _add_tune_runtime(sub.choices["tune"])
    return root


def _subparser(root: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    action = next(item for item in root._actions if isinstance(item, argparse._SubParsersAction))
    return action.choices[name]


# Which prepare flags belong to which runtime. A flag of the other runtime is refused rather than ignored: a
# `--llama-server` beside an NVIDIA prepare, or a `--wheel` beside a native one, is a mistaken command line.
_PREPARE_FLAGS = {"nvidia-container": (("--inference", "inference", True), ("--evaluator-base", "evaluator_base", True),
                                       ("--wheel", "wheel", False), ("--lock", "lock", False)),
                  NATIVE_RUNTIME: (("--llama-server", "llama_server", True),)}


def _check_prepare_arguments(prepare: argparse.ArgumentParser, args) -> None:
    """Runtime-dependent requiredness for ``prepare``, reported the way argparse reports it (usage, exit 2).

    ``--inference``/``--evaluator-base`` cannot be ``required=True`` any more because a native prepare has
    neither; checking them here keeps the NVIDIA command line exactly as strict as it was.
    """
    runtime = args.runtime or DEFAULT_RUNTIME
    missing = [flag for flag, attribute, required in _PREPARE_FLAGS[runtime]
               if required and getattr(args, attribute) is None]
    if missing:
        prepare.error("the following arguments are required: " + ", ".join(missing))
    foreign = [flag for other, flags in _PREPARE_FLAGS.items() if other != runtime
               for flag, attribute, _ in flags if getattr(args, attribute) is not None]
    if foreign:
        prepare.error(f"{', '.join(foreign)} not allowed with --runtime {runtime}")


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


def _add_tune_runtime(tune: argparse.ArgumentParser) -> None:
    """``tune --model ... --runtime metal-native --native-bundle B``: derive a native session.

    Passed to ``session.main_tune`` as ``args.runtime``/``args.native_bundle`` (None when omitted, so an NVIDIA
    command line produces exactly the session it always did). Added beside the session parser, guarded like
    ``--dataset-root``, so it stays a no-op for any flag that parser declares itself.
    """
    for flags, options in ((("--runtime",), {"choices": RUNTIMES, "default": None,
                                              "help": f"the runtime to derive the session for (default "
                                                      f"{DEFAULT_RUNTIME}); --model only. metal-native requires "
                                                      "--native-bundle"}),
                           (("--native-bundle",), {"default": None, "metavar": "PATH",
                                                   "help": "native-bundle.json from `llmbench prepare --runtime "
                                                           "metal-native`; --model with --runtime metal-native "
                                                           "only"})):
        try:
            tune.add_argument(*flags, **options)
        except argparse.ArgumentError:  # the session parser already declares it
            pass


def _default_runner(args):
    """The ``candidate`` runner: dispatches on the config's runtime, building nothing until ``run``.

    An NVIDIA candidate gets exactly ``ContainerRunner(capabilities_dir=..., policy_path=...)`` as before."""
    from .runtime import DispatchRunner
    return DispatchRunner(capabilities_dir=args.capabilities_dir, policy_path=args.policy)


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


def _default_native_preparer(args) -> NativeBundle:
    """A host executable runs only when the persistent policy allows native execution.

    Checked here, before ``native_prep`` is even imported, as well as inside ``prepare_native``: a policy
    written for the NVIDIA containers must never reach code that starts a host binary.
    """
    from ..config import RunMode
    from ..safety import SessionLock
    lock = SessionLock.read(args.policy)
    lock.check("native", RunMode.LIVE)
    from .native_prep import prepare_native
    return prepare_native(Path(args.llama_server), Path(args.output), session_lock=lock,
                          worker_iidfile=args.worker_iidfile)


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
    """Every image the config names must be exactly the prepared one; extra bundle images are fine.

    A metal-native config has no inference image for an image bundle to pin, so comparing would find nothing to
    disagree about and pass; it is reported as a problem instead, and pinned by ``check_native_bundle``."""
    if config.runtime != DEFAULT_RUNTIME:
        return [f"runtime: the config runs {config.runtime}, which an image bundle cannot pin; use its "
                f"{runtime_spec(config.runtime).bundle_name}"]
    problems = _same_image(config.inference_image, bundle.inference, "inference")
    problems += _same_image(config.evaluator.image, bundle.evaluator, "evaluator")
    problems += _same_image(config.worker_image, bundle.worker, "worker")
    # Host evaluation uses the installed registry, checked by host preflight; the bundle records
    # the evaluator image registry and is authoritative only when that image actually evaluates.
    if (config.evaluator.mode == "container" and config.registry_digest is not None
            and config.registry_digest != bundle.registry_digest):
        problems.append("registry_digest: config differs from the bundle's evaluator registry digest")
    return problems


NATIVE_PINS = ("executable_sha256", "libraries_sha256", "build_info", "help_sha256")
"""What identifies a native server. ``executable`` (where it lives on this host) and ``source`` (display text)
are deliberately absent, exactly as they are absent from the config fingerprint."""


def check_native_bundle(config, bundle: NativeBundle) -> list[str]:
    """The native analogue of ``check_image_bundle``: the server the config pins must be exactly the prepared one,
    and so must its sandbox worker image when it names one.

    The bundle's ``registry_digest`` is not compared: a native server is always evaluated by the host process,
    whose installed registry the runner's host preflight checks at admission, just as for a host-process NVIDIA run.
    """
    if config.runtime != NATIVE_RUNTIME or config.native_server is None:
        return [f"runtime: the config runs {config.runtime}; a native bundle pins only {NATIVE_RUNTIME} candidates"]
    problems = []
    for field in NATIVE_PINS:
        configured, bundled = getattr(config.native_server, field), getattr(bundle.native_server, field)
        if configured != bundled:
            problems.append(f"native_server.{field}: config {configured!r} != bundle {bundled!r}")
    return problems + _same_image(config.worker_image, bundle.worker, "worker")


def check_bundle(config, bundle: ImageBundle | NativeBundle) -> list[str]:
    """Either bundle kind against a config, for callers that read whichever bundle a plan names."""
    return (check_native_bundle if isinstance(bundle, NativeBundle) else check_image_bundle)(config, bundle)


def _command_runtime(args, config) -> str:
    """The runtime a pure or candidate command acts for: the config's own, which ``--runtime`` may only confirm."""
    requested = getattr(args, "runtime", None)
    if config is None:
        return requested or DEFAULT_RUNTIME
    if requested is not None and requested != config.runtime:
        raise ValueError(f"--runtime {requested} disagrees with the config, which runs {config.runtime}")
    return config.runtime


def _server_argv(config) -> tuple[str, ...]:
    """The argv the runner will give llama-server: unchanged for NVIDIA; for a native server the host model file,
    loopback and the port placeholder, every setting flag identical."""
    if config.runtime == DEFAULT_RUNTIME:
        return build_server_argv(config)
    return build_server_argv(config, model_path=config.server_model_path, host=NATIVE_HOST,
                             port=NATIVE_PORT_PLACEHOLDER)


def _runtime_report(config) -> dict:
    """The runtime keys a pure command adds for a metal-native config; none for NVIDIA, whose output is unchanged."""
    if config is None or config.runtime == DEFAULT_RUNTIME:
        return {}
    return {"runtime": config.runtime, "required_operations": list(required_operations(config)),
            "unsupported_settings": unsupported_settings(config),
            "not_enforced_settings": not_enforced_settings(config)}


def _check_candidate_bundles(args, config) -> None:
    """Each runtime is pinned by its own bundle; the other kind is refused rather than silently not compared."""
    if config.runtime == NATIVE_RUNTIME:
        if args.image_bundle:
            raise ValueError("--image-bundle pins container images; a metal-native candidate is pinned by "
                             "--native-bundle")
        if args.native_bundle:
            problems = check_native_bundle(config, read_native_bundle(args.native_bundle))
            if problems:
                raise ValueError("config native server differs from the prepared bundle: " + "; ".join(problems))
        return
    if args.native_bundle:
        raise ValueError(f"--native-bundle pins a metal-native server; this config runs {config.runtime} "
                         "(use --image-bundle)")
    if args.image_bundle:
        problems = check_image_bundle(config, read_image_bundle(args.image_bundle))
        if problems:
            raise ValueError("config images differ from the prepared bundle: " + "; ".join(problems))


def _native_bundle_summary(bundle: NativeBundle, output) -> dict:
    server = bundle.native_server
    return {"execution_performed": True, "runtime": bundle.runtime, "output": str(output),
            "executable": server.executable, "native_server": server.executable_sha256,
            "libraries_sha256": server.libraries_sha256, "build_info": server.build_info,
            "worker": bundle.worker.image_id if bundle.worker else None,
            "worker_platform": bundle.worker.platform if bundle.worker else None,
            "sandbox": bundle.sandbox.get("status"), "sandbox_reason": bundle.sandbox.get("reason"),
            "help_sha256": bundle.help_sha256, "registry_digest": bundle.registry_digest,
            "prepared_utc": bundle.prepared_utc}


def _tune_native_bundle(args) -> NativeBundle | None:
    """The native bundle a ``tune --model`` names, or None for an NVIDIA session.

    Both flags are explicit: ``--native-bundle`` never switches the runtime by itself, and ``--runtime
    metal-native`` never runs without the pinned server it describes."""
    runtime = getattr(args, "runtime", None) or DEFAULT_RUNTIME
    path = getattr(args, "native_bundle", None)
    if runtime != NATIVE_RUNTIME:
        if path is not None:
            raise ValueError(f"--native-bundle pins a metal-native server; add --runtime {NATIVE_RUNTIME}")
        return None
    if path is None:
        raise ValueError(f"--runtime {NATIVE_RUNTIME} requires --native-bundle (native-bundle.json from "
                         f"`{PROG} prepare --runtime {NATIVE_RUNTIME}`)")
    if getattr(args, "image_bundle", None):
        raise ValueError("--image-bundle pins container images; a metal-native session is pinned by --native-bundle")
    return read_native_bundle(path)


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

    ``--runtime metal-native --native-bundle B`` is validated here, before the plan is printed, and the bundle is
    handed to the plan exactly as ``main_tune`` hands it to the derivation: it carries the native runtime and
    whether the Docker sandbox was available, which decides whether the coding suites are planned or blocked. An
    NVIDIA command line calls the plan with exactly the arguments it always did.
    """
    from .derive import benchmark_plan_for_model, benchmark_plan_lines
    if args.config:
        if getattr(args, "dataset_root", None) is not None:
            raise ValueError("--dataset-root derives a session from --model; a --config session already states "
                             "its own base.dataset_root")
        if getattr(args, "runtime", None) is not None or getattr(args, "native_bundle", None) is not None:
            raise ValueError("--runtime/--native-bundle derive a session from --model; a --config session already "
                             "states its own base.runtime and server pins")
        return main_tune(args)
    native = _tune_native_bundle(args)
    explicit = getattr(args, "dataset_root", None)
    base = read_run_config(args.base_config)
    plan = benchmark_plan_for_model(args.model, base=base, budget_seconds=args.budget_seconds or 14400,
                                    context_floor=getattr(args, "context_floor", None),
                                    context_ceiling=getattr(args, "context_ceiling", None),
                                    dataset_root=explicit,
                                    **({} if native is None else {"native_bundle": native}))
    if native is not None:
        sandbox = native.sandbox.get("status") or "unrecorded"
        reason = native.sandbox.get("reason")
        print(f"{PROG} tune: runtime {NATIVE_RUNTIME}: llama-server {native.native_server.executable_sha256[:12]} "
              f"({native.native_server.build_info}); coding sandbox {sandbox}" + (f" ({reason})" if reason else ""),
              file=sys.stderr)
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
    root = parser()
    args = root.parse_args(argv)
    if args.command == "prepare":
        _check_prepare_arguments(_subparser(root, "prepare"), args)
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
        runtime = _command_runtime(args, config)  # prepare has no config: its --runtime, else the default
        explicit_capabilities = getattr(args, "capabilities_dir", None) is not None
        if hasattr(args, "capabilities_dir") and not explicit_capabilities:
            args.capabilities_dir = runtime_spec(runtime).default_capabilities_dir
        native = _runtime_report(config)
        if args.command == "validate":
            data = {"valid": not native.get("unsupported_settings"), "label": config.label,
                    "fingerprint": config.fingerprint(), "alias": config.alias(), "execution_performed": False,
                    **native}
            if not data["valid"]:
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return 2
        elif args.command == "capabilities":
            caps = load_capabilities(args.capabilities_dir)
            data = {"flags": len(caps.flags), "help_sha256": caps.help_sha256, "version": caps.version,
                    "build": caps.build, "allowed_flags_missing": sorted(ALLOWED_FLAGS - set(caps.flags)),
                    "execution_performed": False}
            if runtime != DEFAULT_RUNTIME:
                data.update(runtime=runtime, capabilities_dir=str(args.capabilities_dir))
            if config is not None:
                data["help_sha256_matches"] = caps.help_sha256 == config.help_sha256
                data["findings"] = check_argv(_server_argv(config), caps)
                data.update(native)
            if (data["allowed_flags_missing"] or data.get("findings") or data.get("help_sha256_matches") is False
                    or data.get("unsupported_settings")):
                print(json.dumps(data, indent=2))
                return 2
        elif args.command == "plan" and config.runtime != DEFAULT_RUNTIME:
            caps = load_capabilities(args.capabilities_dir)
            argv = _server_argv(config)
            require_supported(argv, caps, expected_help_sha256=config.help_sha256)
            data = {"execution_performed": False,
                    "preview": "the port is a placeholder: the runner starts llama-server on a free loopback port; "
                               "there is no Compose project",
                    "fingerprint": config.fingerprint(), **native, "executable": config.native_server.executable,
                    "server_argv": list(argv)}
            if native["unsupported_settings"]:
                print(json.dumps(data, indent=2, ensure_ascii=False))
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
            require_supported(plan.server_argv, caps, expected_help_sha256=config.help_sha256)
            data = {"execution_performed": False, "preview": "attempt id, RUN_DIR and the evaluator grant are "
                                                             "placeholders",
                    "fingerprint": config.fingerprint(), "server_argv": list(plan.server_argv),
                    "compose": plan.compose}
        elif args.command == "prepare" and runtime == NATIVE_RUNTIME:
            bundle = (preparer or _default_native_preparer)(args)
            if not isinstance(bundle, NativeBundle):
                raise ValueError(f"the native preparer returned {type(bundle).__name__}, not a NativeBundle")
            data = _native_bundle_summary(bundle, args.output)
        elif args.command == "prepare":
            bundle = (preparer or _default_preparer)(args)
            data = {"execution_performed": True, "output": str(args.output),
                    "inference": bundle.inference.image_id, "evaluator": bundle.evaluator.image_id,
                    "worker": bundle.worker.image_id if bundle.worker else None,
                    "help_sha256": bundle.help_sha256, "registry_digest": bundle.registry_digest,
                    "prepared_utc": bundle.prepared_utc}
        else:
            _check_candidate_bundles(args, config)
            if native:
                if explicit_capabilities:
                    raise ValueError("--capabilities-dir does not apply to a metal-native candidate: the native "
                                     "runner reads the executable's own --help and --version at admission")
                if native["unsupported_settings"]:
                    raise ValueError(f"settings the {config.runtime} runtime cannot honour: "
                                     + "; ".join(native["unsupported_settings"]))
            runner = (runner_factory or _default_runner)(args)
            result = runner.run(config, args.output, remaining_budget_seconds=args.budget_seconds)
            dumped = result.model_dump(mode="json")
            data = {"state": result.state, "synthetic": result.synthetic, "attempt_id": result.attempt_id,
                    "failure_stage": result.failure_stage, "failure_reasons": list(result.failure_reasons),
                    "warnings": list(result.warnings), "cleanup_verified": result.cleanup.verified,
                    "abort_campaign": result.abort_campaign, "elapsed_seconds": result.elapsed_seconds,
                    "effective_settings_verified": result.effective_settings_verified,
                    "actual_context_verified": result.actual_context_verified,
                    "minimum_native_tps": result.speed.get("minimum_native_tps"),
                    "samples_total": result.samples_total, "reports": dumped["reports"],
                    "output": str(args.output)}
            if native:  # unified-memory evidence, labelled as such by the runner; never VRAM
                data.update(runtime=config.runtime, memory=dumped.get("memory", {}))
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0 if result.state == "completed" else 4 if result.state == "cleanup-uncertain" else 3
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    except (ValidationError, ValueError, KeyError, OSError, OperationForbidden, UnsupportedSetting) as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
