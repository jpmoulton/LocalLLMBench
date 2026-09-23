"""The inference runtimes a candidate can name, what each one needs and cannot honour, and who runs it.

``ContainerRunConfig.runtime`` says what serves the model; this module is the one place that turns that name into
consequences, so the CLI, the session, the sweeps and the runners cannot disagree about them. Importing performs no
I/O and builds no runner: the runner modules are imported only when a candidate of that runtime actually arrives.

* ``required_operations`` -- the ``runtime-policy.json`` permissions a candidate needs. The NVIDIA tuple is exactly
  the one every live path has always checked. A native server needs ``native`` instead of ``container``, because a
  policy written for the NVIDIA containers must never silently authorize running a host binary; it needs
  ``container`` as well only when the coding broker will start sandboxed Docker workers.
* ``unsupported_settings`` -- settings the schema accepts but the runtime cannot honour. Non-empty means refuse:
  a run that quietly ignored them would report a configuration it never had.
* ``not_enforced_settings`` -- settings that are accepted but bound nothing under that runtime (Docker cgroup
  limits on a host process). They are listed so a report never implies they limited the run.
* ``DispatchRunner`` -- one runner object for either runtime: each candidate is handed to the runner its own
  ``config.runtime`` names, built lazily and exactly as before for NVIDIA.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

from .config import RUNTIMES

__all__ = ["NATIVE_RUNTIME", "RuntimeSpec", "RUNTIME_SPECS", "runtime_spec", "required_operations",
           "unsupported_settings", "not_enforced_settings", "runner_for", "DispatchRunner"]

NATIVE_RUNTIME = "metal-native"


@dataclass(frozen=True)
class RuntimeSpec:
    """Static facts about one runtime. ``memory_kind`` names what a memory number from it measures, so a reader
    never puts an NVIDIA card's dedicated VRAM and an Apple Silicon process footprint in one column."""

    name: str
    description: str
    evaluator_modes: tuple[str, ...]
    memory_kind: Literal["nvidia-vram", "apple-unified"]
    default_capabilities_dir: str
    bundle_name: str


RUNTIME_SPECS: Mapping[str, RuntimeSpec] = MappingProxyType({
    "nvidia-container": RuntimeSpec(
        name="nvidia-container",
        description="pinned llama.cpp CUDA server image run under Docker Compose on an NVIDIA GPU",
        evaluator_modes=("host-process", "container"), memory_kind="nvidia-vram",
        default_capabilities_dir="artifacts/container-prep", bundle_name="image-bundle.json"),
    NATIVE_RUNTIME: RuntimeSpec(
        name=NATIVE_RUNTIME,
        description="pinned llama.cpp llama-server executable run directly on an Apple Silicon macOS host with "
                    "Metal offload; generated code still runs only in the sandboxed Docker worker",
        evaluator_modes=("host-process",), memory_kind="apple-unified",
        default_capabilities_dir="artifacts/native-prep", bundle_name="native-bundle.json"),
})
if tuple(RUNTIME_SPECS) != RUNTIMES:  # a runtime added to the config schema must be described here first
    raise RuntimeError("RUNTIME_SPECS must describe exactly config.RUNTIMES, in order")


def runtime_spec(name: str) -> RuntimeSpec:
    """The spec for a runtime name. An unknown name is refused, never mapped to the default."""
    try:
        return RUNTIME_SPECS[name]
    except (KeyError, TypeError):
        raise ValueError(f"unknown runtime {name!r}; expected one of: {', '.join(RUNTIMES)}") from None


def _runtime_name(runtime: Any) -> str:
    """A runtime name given either as a string or as anything carrying ``.runtime`` (a run config)."""
    name = runtime if isinstance(runtime, str) else getattr(runtime, "runtime", None)
    return runtime_spec(name).name


def required_operations(config) -> tuple[str, ...]:
    """The session-lock operations a live run of ``config`` must be allowed, in the order they are checked.

    NVIDIA: exactly ``("container", "load", "inference")`` whatever else the config says -- the broker's workers
    are containers, already covered. metal-native: ``("native", "load", "inference")``, plus ``"container"`` when
    ``config.broker`` is set, because the coding suites then start sandboxed Docker workers.
    """
    if _runtime_name(config) != NATIVE_RUNTIME:
        return ("container", "load", "inference")
    operations = ("native", "load", "inference")
    return operations + ("container",) if getattr(config, "broker", None) is not None else operations


def unsupported_settings(config) -> list[str]:
    """Settings ``config`` asks for that its runtime cannot honour; empty means none. Always empty for NVIDIA,
    whose runner is the reference every setting was defined against.

    The metal-native list repeats checks the config schema already makes (a container evaluator): this is the
    runtime's own account of what it can do, and it must hold for a config built without validation too.
    """
    if _runtime_name(config) != NATIVE_RUNTIME:
        return []
    problems = []
    if config.evaluator.mode not in RUNTIME_SPECS[NATIVE_RUNTIME].evaluator_modes:
        problems.append(f"evaluator.mode {config.evaluator.mode}: a native server listens on host loopback, which "
                        "the evaluator container's internal network cannot reach; use host-process")
    if config.limits.gpu_device_id != "0":
        problems.append(f"limits.gpu_device_id {config.limits.gpu_device_id!r}: Metal exposes one device, MTL0; "
                        "use \"0\"")
    return problems


def not_enforced_settings(config) -> list[str]:
    """Accepted settings that bound nothing under ``config``'s runtime, each with what applies instead.

    Informational -- a run is not refused for them -- but a report must not present them as limits the run had.
    Always empty for NVIDIA: Docker enforces its cgroup limits and the VRAM admission is its own.
    """
    if _runtime_name(config) != NATIVE_RUNTIME:
        return []
    limits, native = config.limits, config.native_limits
    budget = ("the Metal working-set budget from the startup log" if native is None
              or native.max_server_footprint_mib is None else f"{native.max_server_footprint_mib} MiB")
    gpu_share = ("" if native is None else
                 f" (native_limits.max_foreign_gpu_utilization_percent {native.max_foreign_gpu_utilization_percent})")
    return [
        f"limits.inference_memory_mib ({limits.inference_memory_mib}): a Docker cgroup limit, not enforced for a "
        f"host process; the native_limits admission and memory watchdog apply instead (server footprint ceiling "
        f"{budget})",
        f"limits.inference_cpus ({limits.inference_cpus}): a Docker cgroup limit, not enforced for a host process; "
        "llama-server's CPU use follows engine.threads and engine.threads_batch",
        f"limits.max_foreign_vram_mib ({limits.max_foreign_vram_mib}): NVIDIA-only, there is no dedicated VRAM in "
        f"unified memory; the GPU-utilization admission{gpu_share} replaces it",
    ]


def runner_for(runtime, *, capabilities_dir, policy_path, policy=None,
               native_factory: Callable[..., Any] | None = None,
               container_factory: Callable[..., Any] | None = None):
    """Build the runner for one runtime, named directly or by a config.

    NVIDIA gets ``ContainerRunner(capabilities_dir=..., policy_path=..., policy=...)``: the construction every live
    path used before runtimes existed, so dispatching cannot change its evidence. metal-native gets
    ``NativeRunner(policy_path=..., policy=...)``; it captures the executable's own ``--help`` and ``--version``
    at admission, so it is given no capabilities directory. A factory, when given, is called with exactly those
    keywords in place of the class.
    """
    if _runtime_name(runtime) == NATIVE_RUNTIME:
        if native_factory is None:
            from .native import NativeRunner
            native_factory = NativeRunner
        return native_factory(policy_path=policy_path, policy=policy)
    if container_factory is None:
        from .runner import ContainerRunner
        container_factory = ContainerRunner
    return container_factory(capabilities_dir=capabilities_dir, policy_path=policy_path, policy=policy)


class DispatchRunner:
    """One runner for either runtime: each candidate goes to the runner its own ``config.runtime`` names.

    Construction is inert -- nothing is imported, read or built -- and each runtime's runner is built by
    ``runner_for`` the first time one of its candidates arrives, then reused, just as a session has always run
    every candidate through one runner. The public contract is the runners' own: ``run(config, output_dir, *,
    remaining_budget_seconds=None, lease_held=False)`` returning a ``ContainerRunResult``.

    Dispatch decides nothing else. Admission -- the policy, unsupported settings, the lease, the host checks --
    stays with the delegate, which records a refusal as a rejected attempt with its evidence in the run directory.

    ``synthetic`` is False because this object injects no boundary. Whether a RESULT is synthetic is decided by
    the delegate that produced it and recorded on the result, which is what every report reads.
    """

    synthetic = False

    def __init__(self, *, capabilities_dir, policy_path, policy=None,
                 native_factory: Callable[..., Any] | None = None,
                 container_factory: Callable[..., Any] | None = None) -> None:
        self.capabilities_dir, self.policy_path, self.policy = capabilities_dir, policy_path, policy
        self.native_factory, self.container_factory = native_factory, container_factory
        self._runners: dict[str, Any] = {}

    @property
    def built(self) -> tuple[str, ...]:
        """The runtimes whose runner exists so far, in ``RUNTIMES`` order."""
        return tuple(name for name in RUNTIMES if name in self._runners)

    def runner_for(self, runtime) -> Any:
        """The (lazily built, then reused) runner for a runtime name or a config's runtime."""
        name = _runtime_name(runtime)
        if name not in self._runners:
            self._runners[name] = runner_for(name, capabilities_dir=self.capabilities_dir,
                                             policy_path=self.policy_path, policy=self.policy,
                                             native_factory=self.native_factory,
                                             container_factory=self.container_factory)
        return self._runners[name]

    def run(self, config, output_dir, *, remaining_budget_seconds: float | None = None, lease_held: bool = False):
        return self.runner_for(config).run(config, output_dir, remaining_budget_seconds=remaining_budget_seconds,
                                           lease_held=lease_held)
