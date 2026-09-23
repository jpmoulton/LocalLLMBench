"""The runtime registry: one place that says what each runtime needs, cannot honour and is run by.

The NVIDIA answers must be exactly what every live path did before runtimes existed; the metal-native answers must
never be implied by a container permission, never silently ignore a setting, and never build a runner early.
"""

import inspect
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from llmbench.config import RunMode
from llmbench.containers import cli, image_plan, session
from llmbench.containers.config import (DEFAULT_RUNTIME, RUNTIMES, BrokerSettings, ContainerRunConfig,
                                        EvaluatorSettings, ImageRef, read_run_config)
from llmbench.containers.runner import ContainerRunner
from llmbench.containers.runtime import (RUNTIME_SPECS, DispatchRunner, not_enforced_settings, required_operations,
                                         runner_for, runtime_spec, unsupported_settings)
from llmbench.safety import OperationForbidden, SessionLock

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "candidate.json"
NATIVE_SERVER = {"executable": "/Users/example/llama-b11011/llama-server", "executable_sha256": "e" * 64,
                 "libraries_sha256": "f" * 64, "build_info": "b11011-aa39d7a3e", "help_sha256": "a" * 64,
                 "source": "llama.cpp b11011 macos-arm64 release"}
EVALUATOR_IMAGE = {"role": "evaluator", "reference": "sha256:" + "c" * 64, "image_id": "sha256:" + "c" * 64,
                   "entrypoint": ["python", "-m", "llmbench.container_eval"]}


def native_raw(*, help_sha256: str = "a" * 64, limits: dict | None = None, native_limits: dict | None = None,
               **changes) -> dict:
    """The packaged NVIDIA example turned into a metal-native candidate: same engine, model and benchmarks."""
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw.pop("inference_image")
    raw.update(runtime="metal-native", native_server={**NATIVE_SERVER, "help_sha256": help_sha256},
               native_limits=native_limits or {})
    raw["limits"] = {**raw["limits"], **(limits or {})}
    raw.update(changes)
    return raw


def native_config(**options) -> ContainerRunConfig:
    return ContainerRunConfig.model_validate_json(json.dumps(native_raw(**options)))


def admits(lock: SessionLock, config) -> bool:
    try:
        for operation in required_operations(config):
            lock.check(operation, RunMode.LIVE)
    except OperationForbidden:
        return False
    return True


def test_specs_describe_exactly_the_config_runtimes_and_agree_with_their_consumers():
    assert tuple(RUNTIME_SPECS) == RUNTIMES and DEFAULT_RUNTIME == "nvidia-container"
    nvidia, native = runtime_spec("nvidia-container"), runtime_spec("metal-native")
    # The NVIDIA defaults are the ones the CLI, the session and the runner already used.
    assert nvidia.default_capabilities_dir == cli.DEFAULT_CAPABILITIES == session.DEFAULT_CAPABILITIES
    assert inspect.signature(ContainerRunner).parameters["capabilities_dir"].default == nvidia.default_capabilities_dir
    assert nvidia.bundle_name == image_plan.BUNDLE_NAME
    assert set(nvidia.evaluator_modes) == set(get_args(EvaluatorSettings.model_fields["mode"].annotation))
    # The native ones are what prepare writes and what the native runner reads.
    from llmbench.containers import native_prep
    from llmbench.containers.native import NativeRunner
    assert native.bundle_name == native_prep.NATIVE_BUNDLE_NAME == "native-bundle.json"
    assert Path(native.default_capabilities_dir) == NativeRunner().capabilities_dir == Path("artifacts/native-prep")
    assert native.evaluator_modes == ("host-process",)
    # Memory numbers are labelled by what they measure, so VRAM and a unified-memory footprint never share a column.
    assert (nvidia.memory_kind, native.memory_kind) == ("nvidia-vram", "apple-unified")
    with pytest.raises(FrozenInstanceError):
        nvidia.name = "cuda"
    with pytest.raises(TypeError):
        RUNTIME_SPECS["cuda"] = nvidia
    for bogus in ("cuda", "", None, ["metal-native"], "METAL-NATIVE"):
        with pytest.raises(ValueError, match="unknown runtime"):
            runtime_spec(bogus)


def test_required_operations_keep_the_nvidia_tuple_and_never_derive_native_from_container():
    nvidia = read_run_config(EXAMPLE)
    brokered = nvidia.model_copy(update={"broker": BrokerSettings()})
    assert required_operations(nvidia) == required_operations(brokered) == ("container", "load", "inference")
    native = native_config()
    brokered_native = native.model_copy(update={"broker": BrokerSettings()})
    assert required_operations(native) == ("native", "load", "inference")
    assert required_operations(brokered_native) == ("native", "load", "inference", "container")
    assert required_operations("metal-native") == ("native", "load", "inference")  # a name works as well
    # A policy written for the NVIDIA containers never runs a host binary, and a native-only policy never starts a
    # container: not the NVIDIA server, and not the coding sandbox's workers either.
    base = {"allow_model_operations": True, "allow_inference": True}
    container_only = SessionLock(allow_container_execution=True, **base)
    native_only = SessionLock(allow_native_execution=True, **base)
    both = SessionLock(allow_container_execution=True, allow_native_execution=True, **base)
    assert admits(container_only, nvidia) and not admits(container_only, native)
    assert admits(native_only, native) and not admits(native_only, nvidia) and not admits(native_only, brokered_native)
    assert admits(both, brokered_native)
    assert not admits(SessionLock(allow_native_execution=True, allow_inference=True), native)  # load is still needed
    with pytest.raises(ValueError, match="unknown runtime"):
        required_operations("rocm")


def test_unsupported_settings_refuse_what_metal_cannot_honour_and_never_touch_nvidia():
    nvidia = read_run_config(EXAMPLE)
    second_card = nvidia.model_copy(update={"limits": nvidia.limits.model_copy(update={"gpu_device_id": "1"})})
    assert unsupported_settings(nvidia) == unsupported_settings(second_card) == []
    assert unsupported_settings(native_config()) == []
    [problem] = unsupported_settings(native_config(limits={"gpu_device_id": "1"}))
    assert problem.startswith("limits.gpu_device_id '1'") and "one device, MTL0" in problem
    # The schema already refuses a container evaluator for a native server ...
    with pytest.raises(ValidationError, match="host-process"):
        ContainerRunConfig.model_validate_json(json.dumps(native_raw(
            evaluator={"mode": "container", "image": EVALUATOR_IMAGE})))
    # ... and the runtime's own account refuses it too, for a config that was built without validation.
    container_evaluator = EvaluatorSettings(mode="container",
                                            image=ImageRef.model_validate_json(json.dumps(EVALUATOR_IMAGE)))
    unvalidated = native_config().model_copy(update={"evaluator": container_evaluator})
    assert [item.split(":")[0] for item in unsupported_settings(unvalidated)] == ["evaluator.mode container"]


def test_not_enforced_settings_name_every_docker_and_vram_limit_with_what_replaces_it():
    nvidia = read_run_config(EXAMPLE)
    assert not_enforced_settings(nvidia) == []  # Docker enforces them; the VRAM admission is NVIDIA's own
    notes = not_enforced_settings(native_config(limits={"inference_memory_mib": 8192, "inference_cpus": 4.0,
                                                        "max_foreign_vram_mib": 1024}))
    assert [note.split(" ", 1)[0] for note in notes] == ["limits.inference_memory_mib", "limits.inference_cpus",
                                                          "limits.max_foreign_vram_mib"]
    assert "(8192)" in notes[0] and "not enforced for a host process" in notes[0]
    assert "Metal working-set budget" in notes[0]  # max_server_footprint_mib None: the budget the server logs
    assert "(4.0)" in notes[1] and "engine.threads" in notes[1]
    assert "(1024)" in notes[2] and "NVIDIA-only" in notes[2] and "max_foreign_gpu_utilization_percent 50" in notes[2]
    capped = not_enforced_settings(native_config(native_limits={"max_server_footprint_mib": 4096,
                                                                "max_foreign_gpu_utilization_percent": 20}))
    assert "ceiling 4096 MiB" in capped[0] and "max_foreign_gpu_utilization_percent 20" in capped[2]


class Recorder:
    def __init__(self, kind, **kwargs):
        self.kind, self.kwargs, self.calls = kind, kwargs, []

    def run(self, config, output_dir, *, remaining_budget_seconds=None, lease_held=False):
        self.calls.append((config.runtime, output_dir, remaining_budget_seconds, lease_held))
        return f"{self.kind}-result"


def test_dispatch_is_inert_then_builds_each_runtime_runner_once_with_its_own_arguments(tmp_path):
    built = []

    def factory(kind):
        def make(**kwargs):
            built.append((kind, kwargs))
            return Recorder(kind, **kwargs)
        return make
    dispatch = DispatchRunner(capabilities_dir="caps", policy_path="p.json", policy="campaign-policy",
                              container_factory=factory("container"), native_factory=factory("native"))
    assert dispatch.synthetic is False and dispatch.built == () and built == []
    nvidia, native = read_run_config(EXAMPLE), native_config()
    assert dispatch.run(nvidia, tmp_path / "a", remaining_budget_seconds=60.0, lease_held=True) == "container-result"
    assert dispatch.run(nvidia, tmp_path / "b") == "container-result"
    assert built == [("container", {"capabilities_dir": "caps", "policy_path": "p.json", "policy": "campaign-policy"})]
    assert dispatch.built == ("nvidia-container",)
    assert dispatch.run(native, tmp_path / "c", remaining_budget_seconds=30.0) == "native-result"
    # The native runner captures its own --help at admission, so it is never handed a capabilities directory.
    assert built[1] == ("native", {"policy_path": "p.json", "policy": "campaign-policy"})
    assert dispatch.built == ("nvidia-container", "metal-native") and len(built) == 2
    assert dispatch.runner_for("nvidia-container").calls == [("nvidia-container", tmp_path / "a", 60.0, True),
                                                             ("nvidia-container", tmp_path / "b", None, False)]
    assert dispatch.runner_for(native).calls == [("metal-native", tmp_path / "c", 30.0, False)]
    with pytest.raises(ValueError, match="unknown runtime"):
        dispatch.runner_for("rocm")
    assert len(built) == 2


def test_an_nvidia_dispatch_never_imports_the_native_runner():
    # Checked in a fresh interpreter: deleting the module from this one would leave `llmbench.containers.native`
    # pointing at a second copy of the module for every later test.
    code = ("import sys\n"
            "from llmbench.containers.config import read_run_config\n"
            "from llmbench.containers.runtime import DispatchRunner\n"
            "dispatch = DispatchRunner(capabilities_dir='caps', policy_path='p.json')\n"
            "assert 'llmbench.containers.native' not in sys.modules, 'imported by construction'\n"
            f"dispatch.runner_for(read_run_config({str(EXAMPLE)!r}))\n"
            "assert 'llmbench.containers.native' not in sys.modules, 'imported by an NVIDIA dispatch'\n"
            "print(dispatch.built)\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "('nvidia-container',)"


def test_default_delegates_are_the_real_runners_constructed_exactly_as_before(tmp_path):
    from llmbench.containers import runner as runner_module
    dispatch = DispatchRunner(capabilities_dir=tmp_path, policy_path=tmp_path / "p.json")
    container = dispatch.runner_for(read_run_config(EXAMPLE))
    reference = ContainerRunner(capabilities_dir=tmp_path, policy_path=tmp_path / "p.json")
    assert type(container) is ContainerRunner
    for attribute in ("synthetic", "executor", "capabilities_dir", "policy_path", "policy", "coding_hook",
                      "lease_path", "session_lock", "hasher"):
        assert getattr(container, attribute) == getattr(reference, attribute), attribute
    assert container.broker_factory is reference.broker_factory is runner_module._default_broker_factory
    from llmbench.containers.native import NativeRunner
    native = runner_for("metal-native", capabilities_dir="ignored", policy_path=tmp_path / "p.json", policy=None)
    assert type(native) is NativeRunner and native.synthetic is False and native.policy_path == tmp_path / "p.json"
    assert dispatch.runner_for(native_config()) is not container and dispatch.built == RUNTIMES
