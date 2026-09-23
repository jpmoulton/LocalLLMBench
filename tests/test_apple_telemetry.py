"""Apple unified-memory telemetry (package A).

The `apple-*.txt` fixtures are verbatim captures of the exact read-only commands the module runs, taken on the
2020 M1 MacBook Pro (8 GB) this runtime was developed on: at rest on battery, on AC with the battery held
("AC attached; not charging"), and the IOAccelerator dump both idle and with Qwen3-1.7B-Q4_K_M resident on Metal.
Everything else is driven through injected runners, readers, samplers and clocks; the only live probes are the
darwin-only libproc/psutil checks of this test process itself.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llmbench import apple
from llmbench.apple import (MIB, MemoryWatchdog, check_unified_memory_headroom, host_facts, parse_ioreg_accelerator,
                            parse_pmset_batt, parse_pressure_level, parse_swapusage, parse_vm_stat,
                            process_alive, process_phys_footprint, process_rss, sample_unified_memory)
from llmbench.containers.config import NativeLimits
from llmbench.containers.preflight import PreflightError

DATA = Path(__file__).parent / "data"
DARWIN_ONLY = pytest.mark.skipif(sys.platform != "darwin", reason="libproc phys_footprint exists only on macOS")

FIXTURES = {
    ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "apple-sysctl-memorystatus-vm-pressure-level.txt",
    ("sysctl", "-n", "vm.swapusage"): "apple-sysctl-vm-swapusage.txt",
    ("vm_stat",): "apple-vm-stat.txt",
    ("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"): "apple-ioreg-accelerator.txt",
    ("pmset", "-g", "batt"): "apple-pmset-batt-discharging.txt",
    ("sysctl", "-n", "machdep.cpu.brand_string"): "apple-sysctl-machdep-cpu-brand-string.txt",
    ("sysctl", "-n", "hw.memsize"): "apple-sysctl-hw-memsize.txt",
    ("sysctl", "-n", "hw.perflevel0.physicalcpu"): "apple-sysctl-hw-perflevel0-physicalcpu.txt",
    ("sysctl", "-n", "hw.perflevel1.physicalcpu"): "apple-sysctl-hw-perflevel1-physicalcpu.txt",
}
SAMPLE_ARGV = [["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], ["sysctl", "-n", "vm.swapusage"],
               ["vm_stat"], ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], ["pmset", "-g", "batt"]]


def fixture(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


class FakeRunner:
    """subprocess.run stand-in: serves the captured output for each exact argv, or raises/returns what a test
    overrides. Any argv not in the table is a test failure, which pins the exact command shapes."""

    def __init__(self, overrides=None):
        self.outputs = {argv: fixture(name) for argv, name in FIXTURES.items()}
        self.outputs.update(overrides or {})
        self.calls = []

    def __call__(self, argv, **kwargs):
        assert isinstance(argv, list), "argv must be a list, never a shell string"
        self.calls.append((argv, kwargs))
        outcome = self.outputs[tuple(argv)]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, subprocess.CompletedProcess):
            return outcome
        return subprocess.CompletedProcess(argv, 0, stdout=outcome, stderr="")


class VirtualMemory:
    total, available, used = 8 * 1024 * MIB, 2103 * MIB, 3642 * MIB


def fake_readers(**changes):
    readers = {"footprint_reader": lambda pid: 1905 * MIB, "rss_reader": lambda pid: 240 * MIB,
               "alive_reader": lambda pid: True, "virtual_memory": lambda: VirtualMemory(),
               "clock": iter([100.0, 100.05]).__next__}
    readers.update(changes)
    return readers


# ------------------------------------------------------------------------------------------ parsers on captures


def test_ioreg_idle_capture_is_parsed_exactly():
    gpu = parse_ioreg_accelerator(fixture("apple-ioreg-accelerator.txt"))
    assert gpu == {"device_utilization_percent": 34, "renderer_utilization_percent": 33,
                   "tiler_utilization_percent": 34, "in_use_system_memory_bytes": 200048640,
                   "alloc_system_memory_bytes": 2330050560, "gpu_core_count": 8, "model": "Apple M1"}


def test_ioreg_capture_with_a_resident_metal_model():
    gpu = parse_ioreg_accelerator(fixture("apple-ioreg-accelerator-after-load.txt"))
    # Driver in-use memory is 1920 MiB with the model resident, ~1730 MiB above the idle capture: room for the
    # 1631 MiB of Metal buffers the server logged (model 1050 + KV 532 + compute 48) plus other apps' drift.
    assert gpu["in_use_system_memory_bytes"] == 2013724672 and gpu["alloc_system_memory_bytes"] == 4134993920
    assert gpu["device_utilization_percent"] == 33 and gpu["gpu_core_count"] == 8


def test_ioreg_keys_match_exactly_and_the_gpu_is_never_guessed():
    entry = ('+-o AGXAcceleratorG13G_B0  <class AGXAcceleratorG13G_B0>\n    {\n'
             '      "PerformanceStatistics" = {"In use system memory (driver)"=999,"In use system memory"=5,'
             '"Device Utilization %"=7}\n      "model" = "Apple M1"\n    }\n')
    gpu = parse_ioreg_accelerator(entry)
    assert gpu["in_use_system_memory_bytes"] == 5 and gpu["device_utilization_percent"] == 7
    # Keys the driver did not print are None, not zero.
    assert gpu["renderer_utilization_percent"] is None and gpu["gpu_core_count"] is None
    with pytest.raises(ValueError, match="exactly one"):
        parse_ioreg_accelerator(entry + entry)
    with pytest.raises(ValueError, match="0 IOAccelerator"):
        parse_ioreg_accelerator("+-o AGXAccelerator  <class AGXAccelerator>\n    {\n      \"model\" = \"x\"\n    }\n")
    with pytest.raises(ValueError, match="repeats"):
        parse_ioreg_accelerator(entry.replace('"In use system memory"=5', '"Device Utilization %"=5'))
    # A nonsense reading is refused, never compared to a limit: "-1 %" would otherwise look like an idle GPU.
    for bad in ('"Device Utilization %"=-1', '"Device Utilization %"=101'):
        with pytest.raises(ValueError, match="device_utilization_percent = .* is out of range"):
            parse_ioreg_accelerator(entry.replace('"Device Utilization %"=7', bad))
    with pytest.raises(ValueError, match="in_use_system_memory_bytes = -5 is out of range"):
        parse_ioreg_accelerator(entry.replace('"In use system memory"=5', '"In use system memory"=-5'))


def test_pmset_captures_battery_and_ac_attached():
    assert parse_pmset_batt(fixture("apple-pmset-batt-discharging.txt")) == {
        "power_source": "battery", "power_source_label": "Battery Power", "battery_percent": 72,
        "battery_state": "discharging", "charging": False}
    assert parse_pmset_batt(fixture("apple-pmset-batt-ac-attached.txt")) == {
        "power_source": "ac", "power_source_label": "AC Power", "battery_percent": 69,
        "battery_state": "AC attached", "charging": False}


@pytest.mark.parametrize(("text", "expected"), [
    ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t41%; charging; 1:10 remaining present: true\n",
     ("ac", 41, True)),
    ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n",
     ("ac", 100, False)),
    ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; mystery state; x present: true\n",
     ("ac", 80, None)),  # an unknown state is recorded, not mapped by resemblance
    ("Now drawing from 'AC Power'\n", ("ac", None, None)),  # a desktop Mac: no battery at all
    ("Now drawing from 'UPS Power'\n", (None, None, None)),  # not a source this module names
])
def test_pmset_synthetic_variants(text, expected):
    power = parse_pmset_batt(text)
    assert (power["power_source"], power["battery_percent"], power["charging"]) == expected


def test_pmset_unrecognised_output_is_an_error_not_an_absent_battery():
    with pytest.raises(ValueError, match="power source"):
        parse_pmset_batt("")
    with pytest.raises(ValueError, match="battery line"):
        parse_pmset_batt("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\tno percentage here\n")
    with pytest.raises(ValueError, match="more than one"):
        parse_pmset_batt(fixture("apple-pmset-batt-discharging.txt") + " -InternalBattery-1 (id=2)\t5%; charging;\n")


def test_vm_stat_capture_is_parsed_into_ints():
    vm = parse_vm_stat(fixture("apple-vm-stat.txt"))
    assert vm["page_size_bytes"] == 16384 and vm["pages_free"] == 3921
    assert (vm["swapins"], vm["swapouts"], vm["pageouts"]) == (10976953, 12652528, 5458717)
    assert vm["translation_faults"] == 6715043770 and vm["pages_copy_on_write"] == 32115350
    assert vm["file_backed_pages"] == 78134 and vm["pages_occupied_by_compressor"] == 113352
    assert all(type(value) is int for value in vm.values()) and len(vm) == 23


def test_vm_stat_unknown_shapes_are_errors():
    text = fixture("apple-vm-stat.txt")
    with pytest.raises(ValueError, match="page-size header"):
        parse_vm_stat(text.split("\n", 1)[1])
    with pytest.raises(ValueError, match="unrecognised vm_stat line"):
        parse_vm_stat(text + "Pages frobbed:   twelve.\n")
    with pytest.raises(ValueError, match="swapins"):
        parse_vm_stat("\n".join(line for line in text.splitlines() if not line.startswith("Swapins")))


def test_swapusage_capture_is_converted_to_bytes():
    swap = parse_swapusage(fixture("apple-sysctl-vm-swapusage.txt"))
    assert swap == {"swap_total_bytes": 9216 * MIB, "swap_used_bytes": round(7798.56 * MIB),
                    "swap_free_bytes": round(1417.44 * MIB), "swap_encrypted": True}
    assert parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M  ")["swap_encrypted"] is False
    with pytest.raises(ValueError, match="lacks used"):
        parse_swapusage("total = 1024.00M  free = 1024.00M")


def test_localized_swapusage_is_refused_and_probes_run_in_the_c_locale():
    # Captured with LC_ALL=de_DE.UTF-8 on the development M1: sysctl honours the numeric locale. The parser must
    # refuse it (never read "9216,00M" as 9216 or 00 MiB), and the probes must never see the caller's locale.
    with pytest.raises(ValueError, match="lacks total, used, free"):
        parse_swapusage("total = 9216,00M  used = 7742,00M  free = 1474,00M  (encrypted)")
    runner = FakeRunner()
    sample_unified_memory(command_runner=runner, include_gpu=False, include_power=False, **fake_readers())
    assert all(kwargs["env"] == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"} for _, kwargs in runner.calls)
    # Each call gets its own copy: a runner that mutates env cannot change the next probe's environment.
    assert len({id(kwargs["env"]) for _, kwargs in runner.calls}) == len(runner.calls)
    assert apple.PROBE_ENVIRONMENT == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}


def test_pressure_level_vocabulary_is_closed():
    assert parse_pressure_level(fixture("apple-sysctl-memorystatus-vm-pressure-level.txt")) == 1
    assert parse_pressure_level("4\n") == 4
    for text in ("3", "0", "", "warn", "-1"):
        with pytest.raises(ValueError, match="pressure level"):
            parse_pressure_level(text)


# ------------------------------------------------------------------------------------------------------- samples


def test_sample_runs_exactly_the_five_commands_bounded_and_labels_every_value():
    runner = FakeRunner()
    sample = sample_unified_memory(4242, command_runner=runner, timeout=10.0, **fake_readers())
    assert [argv for argv, _ in runner.calls] == SAMPLE_ARGV
    for _, kwargs in runner.calls:  # a caller's 10 s is clamped: no probe may run longer than 3 s
        assert kwargs == {"capture_output": True, "text": True, "timeout": 3.0, "check": True,
                          "env": {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}}
    assert sample == {
        "kind": "apple-unified", "monotonic_seconds": 100.0, "host_memory_total_bytes": 8 * 1024 * MIB,
        "host_memory_available_bytes": 2103 * MIB, "host_memory_used_bytes": 3642 * MIB,
        "swap_total_bytes": 9216 * MIB, "swap_used_bytes": round(7798.56 * MIB), "swapins": 10976953,
        "swapouts": 12652528, "memory_pressure_level": 1,
        "process": {"pid": 4242, "alive": True, "phys_footprint_bytes": 1905 * MIB, "rss_bytes": 240 * MIB},
        "gpu": parse_ioreg_accelerator(fixture("apple-ioreg-accelerator.txt")),
        "power": parse_pmset_batt(fixture("apple-pmset-batt-discharging.txt")),
        "errors": [], "sample_seconds": pytest.approx(0.05)}
    json.dumps(sample, allow_nan=False)
    # Unified memory is never dressed up as VRAM.
    assert "vram" not in json.dumps(sample).lower()


def test_sample_honours_a_shorter_timeout_and_skips_disabled_probes():
    runner = FakeRunner()
    sample = sample_unified_memory(command_runner=runner, timeout=0.5, include_gpu=False, include_power=False,
                                   **fake_readers())
    assert [argv for argv, _ in runner.calls] == SAMPLE_ARGV[:3]
    assert {kwargs["timeout"] for _, kwargs in runner.calls} == {0.5}
    # Not attempted is not a failure: no error, and no process block without a pid.
    assert sample["gpu"] is None and sample["power"] is None and sample["process"] is None
    assert sample["errors"] == []


def test_sample_records_every_probe_failure_and_never_raises():
    runner = FakeRunner({
        ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): subprocess.CalledProcessError(1, ["sysctl"]),
        ("sysctl", "-n", "vm.swapusage"): "garbage",
        ("vm_stat",): FileNotFoundError(2, "No such file or directory", "vm_stat"),
        ("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"): subprocess.TimeoutExpired(["ioreg"], 3.0),
        ("pmset", "-g", "batt"): "x" * (apple.MAX_COMMAND_OUTPUT_CHARS + 1),
    })

    def broken_footprint(pid):
        raise OSError("libproc unavailable")

    def broken_memory():  # how psutil reports a failed host_statistics64 on macOS
        raise RuntimeError("host_statistics(HOST_VM_INFO) syscall failed")

    sample = sample_unified_memory(7, command_runner=runner, **fake_readers(
        footprint_reader=broken_footprint, virtual_memory=broken_memory))
    for key in ("host_memory_available_bytes", "memory_pressure_level", "swap_used_bytes", "swapins", "gpu",
                "power"):
        assert sample[key] is None, key
    assert sample["process"] == {"pid": 7, "alive": True, "phys_footprint_bytes": None, "rss_bytes": 240 * MIB}
    labels = [error.split(":")[0] for error in sample["errors"]]
    assert labels == ["host memory", "memory pressure", "swap usage", "vm_stat", "process phys_footprint_bytes",
                      "gpu", "power"]
    assert "TimeoutExpired" in sample["errors"][5] and "exceeds" in sample["errors"][6]
    json.dumps(sample, allow_nan=False)


def test_an_unreadable_footprint_of_a_running_server_is_an_error_not_a_silent_none():
    sample = sample_unified_memory(99, command_runner=FakeRunner(), include_gpu=False, include_power=False,
                                   **fake_readers(footprint_reader=lambda pid: None))
    assert sample["process"]["phys_footprint_bytes"] is None
    assert sample["errors"] == ["process phys_footprint_bytes: unreadable for a running process"]
    # A process that has exited has no footprint to read: that None is the truth, not a failure.
    gone = sample_unified_memory(99, command_runner=FakeRunner(), include_gpu=False, include_power=False,
                                 **fake_readers(footprint_reader=lambda pid: None, alive_reader=lambda pid: False))
    assert gone["process"]["alive"] is False and gone["errors"] == []


def test_a_runner_that_ignores_check_still_fails_closed():
    failed = subprocess.CompletedProcess(["sysctl"], 1, stdout="4\n", stderr="denied")
    runner = FakeRunner({("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): failed})
    sample = sample_unified_memory(command_runner=runner, include_gpu=False, include_power=False, **fake_readers())
    assert sample["memory_pressure_level"] is None
    assert sample["errors"][0].startswith("memory pressure: CalledProcessError")
    # A result without an exit status is not a success either (no silent default of 0).
    statusless = subprocess.CompletedProcess(["sysctl"], None, stdout="1\n", stderr="")
    runner = FakeRunner({("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): statusless})
    sample = sample_unified_memory(command_runner=runner, include_gpu=False, include_power=False, **fake_readers())
    assert sample["memory_pressure_level"] is None
    assert sample["errors"] == ["memory pressure: ValueError: probe runner reported no integer exit status"]


def test_missing_psutil_leaves_host_memory_unmeasured(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # `import psutil` now raises ImportError
    assert process_rss(os.getpid()) is None and process_alive(os.getpid()) is None
    sample = sample_unified_memory(os.getpid(), command_runner=FakeRunner(), footprint_reader=lambda pid: 1)
    assert sample["host_memory_available_bytes"] is None and sample["process"]["rss_bytes"] is None
    assert sample["errors"][0].startswith("host memory: ModuleNotFoundError")


def test_invalid_arguments_are_caller_bugs_not_probe_failures():
    for timeout in (0, -1, float("nan"), float("inf"), True, "3"):
        with pytest.raises(ValueError, match="timeout"):
            sample_unified_memory(command_runner=FakeRunner(), timeout=timeout)
    for pid in (0, -5, True, "12", 2**31):  # pid_t is 32-bit: a larger int would be a ctypes error, not None
        with pytest.raises(ValueError, match="pid"):
            sample_unified_memory(pid, command_runner=FakeRunner())
        with pytest.raises(ValueError, match="pid"):
            process_phys_footprint(pid)


def test_host_facts_are_stable_facts_only(monkeypatch):
    monkeypatch.setattr(apple.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(apple.platform, "mac_ver", lambda: ("14.2.1", ("", "", ""), "arm64"))
    runner = FakeRunner()
    facts = host_facts(command_runner=runner)
    assert facts == {"machine": "arm64", "chip": "Apple M1", "memory_bytes": 8589934592, "macos": "14.2.1",
                     "performance_cores": 4, "efficiency_cores": 4, "gpu_core_count": 8, "errors": []}
    # Nothing that changes second to second: no pressure, swap, vm_stat or power probe.
    assert {tuple(argv) for argv, _ in runner.calls}.isdisjoint(
        {("sysctl", "-n", "kern.memorystatus_vm_pressure_level"), ("sysctl", "-n", "vm.swapusage"), ("vm_stat",),
         ("pmset", "-g", "batt")})
    assert all(kwargs["timeout"] <= 3.0 for _, kwargs in runner.calls)


def test_host_facts_on_a_mac_without_efficiency_cores(monkeypatch):
    monkeypatch.setattr(apple.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    missing = subprocess.CalledProcessError(1, ["sysctl"], stderr="unknown oid 'hw.perflevel1.physicalcpu'")
    facts = host_facts(command_runner=FakeRunner({("sysctl", "-n", "hw.perflevel1.physicalcpu"): missing}))
    assert facts["efficiency_cores"] is None and facts["macos"] is None and facts["performance_cores"] == 4
    assert len(facts["errors"]) == 1 and facts["errors"][0].startswith("efficiency cores: CalledProcessError")


def test_importing_the_module_probes_nothing():
    # In a fresh interpreter, so the module really is imported (not reloaded) with every probe path booby-trapped.
    code = ("import ctypes, subprocess\n"
            "def forbidden(*args, **kwargs):\n"
            "    raise SystemExit('importing llmbench.apple ran a probe')\n"
            "subprocess.run = subprocess.Popen = ctypes.CDLL = forbidden\n"
            "import llmbench.apple as apple\n"
            "assert apple._libproc.cache_info().currsize == 0\n"
            "print('inert')\n")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                               cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONPATH": "src"})
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "inert"


def test_the_nvidia_container_path_never_imports_apple_telemetry():
    code = ("import sys, llmbench.telemetry, llmbench.containers.runner, llmbench.containers.preflight; "
            "print('llmbench.apple' in sys.modules)")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                               cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONPATH": "src"})
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


# ----------------------------------------------------------------------------------------------------- admission


def unified(available_mib=4096, pressure=1, gpu=20, swap_mib=6000, **changes):
    sample = {"kind": "apple-unified", "monotonic_seconds": 1.0, "host_memory_available_bytes": available_mib * MIB,
              "memory_pressure_level": pressure, "swap_used_bytes": swap_mib * MIB, "swapins": 10, "swapouts": 20,
              "gpu": None if gpu is None else {"device_utilization_percent": gpu},
              "power": {"power_source": "battery", "battery_percent": 72, "charging": False}, "errors": []}
    sample.update(changes)
    return sample


def test_headroom_admits_with_evidence():
    samples = [unified(4096, gpu=20), unified(3000, gpu=90), unified(3500, gpu=30, swap_mib=6100)]
    evidence = check_unified_memory_headroom(samples, required_bytes=2500 * MIB, limits=NativeLimits())
    # Worst memory reading, median GPU (one 90 % spike does not refuse), latest swap and power.
    assert evidence == {"kind": "apple-unified", "available_bytes": 3000 * MIB, "required_bytes": 2500 * MIB,
                        "pressure_level": 1, "max_pressure_level": 2, "gpu_utilization_percent_samples": [20, 90, 30],
                        "gpu_utilization_percent_median": 30, "max_foreign_gpu_utilization_percent": 50,
                        "swap_used_bytes": 6100 * MIB, "power": samples[-1]["power"], "samples": 3,
                        "verdict": "admitted"}
    json.dumps(evidence, allow_nan=False)


def test_headroom_refuses_insufficient_memory_pressure_and_a_busy_gpu_together():
    samples = [unified(1500, pressure=2, gpu=70), unified(1400, pressure=4, gpu=65)]
    with pytest.raises(PreflightError) as caught:
        check_unified_memory_headroom(samples, required_bytes=2500 * MIB, limits=NativeLimits())
    message = str(caught.value)
    assert "1400 MiB of unified memory available" in message and "needs 2500 MiB" in message
    assert "pressure is level 4 (critical)" in message and "limit for admission is 2" in message
    assert "GPU device utilization is already 67.5% (median of 2 samples)" in message
    assert "max_foreign_gpu_utilization_percent is 50" in message
    assert message.endswith("A competing workload is a conflict to report, never a process to stop")


def test_headroom_pressure_limit_comes_from_native_limits():
    samples = [unified(pressure=2)]
    assert check_unified_memory_headroom(samples, required_bytes=0, limits=NativeLimits())["pressure_level"] == 2
    with pytest.raises(PreflightError, match="pressure is level 2 .warn."):
        check_unified_memory_headroom(samples, required_bytes=0, limits=NativeLimits(max_memory_pressure_level=1))


def test_headroom_refuses_without_evidence():
    limits = NativeLimits()
    with pytest.raises(PreflightError, match="telemetry is unavailable.*psutil missing"):
        check_unified_memory_headroom([unified(host_memory_available_bytes=None, errors=["psutil missing"])],
                                      required_bytes=0, limits=limits)
    with pytest.raises(PreflightError, match="telemetry is unavailable"):
        check_unified_memory_headroom([], required_bytes=0, limits=limits)
    # A sample of another kind (an NVIDIA telemetry row) is not evidence about unified memory.
    with pytest.raises(PreflightError, match="telemetry is unavailable"):
        check_unified_memory_headroom([unified(kind="nvidia")], required_bytes=0, limits=limits)
    with pytest.raises(PreflightError, match="GPU utilization telemetry is unavailable"):
        check_unified_memory_headroom([unified(gpu=None), unified(gpu=None)], required_bytes=0, limits=limits)
    with pytest.raises(ValueError, match="required_bytes"):
        check_unified_memory_headroom([unified()], required_bytes=-1, limits=limits)


def test_headroom_over_real_sample_shape():
    sample = sample_unified_memory(1, command_runner=FakeRunner(), **fake_readers())
    evidence = check_unified_memory_headroom([sample], required_bytes=1024 * MIB, limits=NativeLimits())
    assert evidence["gpu_utilization_percent_samples"] == [34] and evidence["power"]["power_source"] == "battery"


# ------------------------------------------------------------------------------------------------------ watchdog


class Sequence:
    """A sampler that replays scripted samples, then repeats the last one."""

    def __init__(self, *samples):
        self.samples, self.calls = list(samples), []

    def __call__(self, pid):
        self.calls.append(pid)
        return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]


def watched(footprint_mib=None, swap_mib=6000, pressure=1, gpu=30, available_mib=3000, t=1.0, rss_mib=200,
            swapins=100, swapouts=200):
    return unified(available_mib, pressure=pressure, gpu=gpu, swap_mib=swap_mib, monotonic_seconds=t,
                   swapins=swapins, swapouts=swapouts,
                   process={"pid": 42, "alive": True, "rss_bytes": rss_mib * MIB,
                            "phys_footprint_bytes": None if footprint_mib is None else footprint_mib * MIB})


@pytest.fixture
def no_kill(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the watchdog must never signal a process itself")

    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)


def test_watchdog_summary_aggregates_every_sample(no_kill):
    sampler = Sequence(watched(1800, 6000, gpu=30, t=1, swapins=100, swapouts=200, available_mib=3000),
                       watched(1905, 6100, gpu=40, t=2, swapins=150, swapouts=260, available_mib=2500, rss_mib=250),
                       watched(1850, 6050, pressure=2, gpu=20, t=3, swapins=170, swapouts=300, available_mib=2800))
    reasons = []
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), budget_bytes=5461 * MIB,
                         on_violation=reasons.append, sampler=sampler)
    for _ in range(3):
        dog.sample_once()
    summary = dog.stop()
    assert sampler.calls == [42, 42, 42] and reasons == []
    assert summary["sample_count"] == 3 and summary["samples"] == dog.samples() and len(summary["samples"]) == 3
    assert summary["first"]["monotonic_seconds"] == 1 and summary["last"]["monotonic_seconds"] == 3
    assert summary["peak_phys_footprint_bytes"] == 1905 * MIB and summary["peak_rss_bytes"] == 250 * MIB
    assert summary["min_available_bytes"] == 2500 * MIB and summary["max_pressure_level"] == 2
    assert (summary["swap_used_start_bytes"], summary["swap_used_peak_bytes"], summary["swap_used_end_bytes"]) == (
        6000 * MIB, 6100 * MIB, 6050 * MIB)
    assert summary["swap_growth_bytes"] == 100 * MIB and summary["swap_baseline"] == "first-sample"
    assert (summary["swapins_delta"], summary["swapouts_delta"]) == (70, 100)
    assert summary["gpu_utilization_percent"] == {"median": 30, "median_of_last": 3, "max": 40, "samples": 3}
    assert summary["footprint_limit_bytes"] == 5461 * MIB and summary["footprint_limit_source"] == "metal_budget"
    assert summary["violations"] == [] and summary["errors"] == [] and summary["started"] is False
    json.dumps(summary, allow_nan=False)


def test_watchdog_footprint_limit_prefers_the_configured_cap_over_the_metal_budget(no_kill):
    reasons = []
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(max_server_footprint_mib=1900),
                         budget_bytes=5461 * MIB, on_violation=reasons.append,
                         sampler=Sequence(watched(1800), watched(1905, t=2.0)))
    dog.sample_once()
    assert reasons == []
    dog.sample_once()
    assert reasons == ["server phys_footprint 1905 MiB exceeds the 1900 MiB limit (max_server_footprint_mib)"]
    assert dog.summary()["violations"] == [{"kind": "footprint", "reason": reasons[0], "monotonic_seconds": 2.0}]


def test_watchdog_uses_the_metal_budget_or_no_footprint_rule(no_kill):
    reasons = []
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), budget_bytes=1500 * MIB,
                         on_violation=reasons.append, sampler=Sequence(watched(1905)))
    dog.sample_once()
    assert reasons == ["server phys_footprint 1905 MiB exceeds the 1500 MiB limit (metal_budget)"]
    unbounded = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), on_violation=reasons.append,
                               sampler=Sequence(watched(100_000)))
    unbounded.sample_once()
    assert len(reasons) == 1 and unbounded.summary()["footprint_limit_source"] == "none"


def test_watchdog_swap_growth_is_measured_from_the_admission_baseline(no_kill):
    reasons = []
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(max_swap_growth_mib=512),
                         on_violation=reasons.append, baseline=unified(swap_mib=5000),
                         sampler=Sequence(watched(swap_mib=5400), watched(swap_mib=5600, t=2.0)))
    dog.sample_once()
    assert reasons == []
    dog.sample_once()
    assert reasons == ["host swap in use grew by 600 MiB; the limit is 512 MiB (swap is host-wide)"]
    summary = dog.summary()
    assert summary["swap_baseline"] == "baseline" and summary["swap_growth_bytes"] == 600 * MIB
    assert summary["swapins_delta"] == 90  # baseline 10 -> 100


def test_watchdog_baseline_alone_measures_no_growth(no_kill):
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), on_violation=lambda reason: None,
                         baseline=unified(swap_mib=5000),
                         sampler=Sequence(watched(100) | {"swap_used_bytes": None}))
    dog.sample_once()
    assert dog.summary()["swap_growth_bytes"] is None


def test_watchdog_reports_only_the_first_violation_but_records_all(no_kill):
    reasons = []
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(max_server_footprint_mib=1024),
                         on_violation=reasons.append,
                         sampler=Sequence(watched(512, swap_mib=100), watched(2048, swap_mib=100, pressure=4, t=2.0),
                                          watched(2048, swap_mib=9000, pressure=4, t=3.0)))
    for _ in range(4):
        dog.sample_once()
    kinds = [(item["kind"], item["monotonic_seconds"]) for item in dog.summary()["violations"]]
    assert kinds == [("footprint", 2.0), ("pressure_critical", 2.0), ("swap_growth", 3.0)]
    assert reasons == ["server phys_footprint 2048 MiB exceeds the 1024 MiB limit (max_server_footprint_mib)"]


def test_watchdog_keeps_bounded_samples_but_exact_peaks(no_kill):
    footprints = [100, 900, 300, 200, 250, 260, 270, 280, 290, 295]
    gpus = [10, 95, 20, 20, 20, 20, 20, 30, 40, 50]  # the GPU spike, like the footprint peak, is in a dropped sample
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), on_violation=lambda reason: None, max_samples=3,
                         sampler=Sequence(*[watched(mib, gpu=gpu, t=float(index))
                                            for index, (mib, gpu) in enumerate(zip(footprints, gpus))]))
    for _ in footprints:
        dog.sample_once()
    summary = dog.summary()
    assert summary["sample_count"] == 10 and summary["samples_retained"] == 3 and len(dog.samples()) == 3
    assert summary["peak_phys_footprint_bytes"] == 900 * MIB and summary["first"]["monotonic_seconds"] == 0.0
    assert [sample["monotonic_seconds"] for sample in summary["samples"]] == [7.0, 8.0, 9.0]
    assert summary["gpu_utilization_percent"] == {"median": 40, "median_of_last": 3, "max": 95, "samples": 10}


def test_watchdog_failures_become_summary_errors(no_kill):
    outcomes = iter([RuntimeError("ioreg wedged"), "not a dict", watched(4096, t=3.0)])

    def sampler(pid):
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def angry(reason):
        raise RuntimeError("runner could not stop the server")

    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(max_server_footprint_mib=1024), on_violation=angry,
                         sampler=sampler)
    assert dog.sample_once() is None and dog.sample_once() is None
    assert dog.sample_once()["monotonic_seconds"] == 3.0
    summary = dog.summary()
    assert summary["errors"] == ["sampler: RuntimeError: ioreg wedged", "sampler returned str, not a sample",
                                 "on_violation: RuntimeError: runner could not stop the server"]
    assert summary["sample_count"] == 1 and [item["kind"] for item in summary["violations"]] == ["footprint"]


def test_watchdog_sample_errors_are_bounded_and_deduplicated(no_kill):
    samples = [watched(100, t=float(index)) | {"errors": [f"gpu: TimeoutExpired {index % 60}"]} for index in range(120)]
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), on_violation=lambda reason: None,
                         sampler=Sequence(*samples))
    for _ in samples:
        dog.sample_once()
    summary = dog.summary()
    # 60 distinct messages, each seen twice: the first 50 are kept (repeats deduplicated), and every occurrence
    # of the other 10 is counted as dropped rather than silently lost.
    assert summary["samples_with_errors"] == 120 and len(summary["errors"]) == apple.MAX_WATCHDOG_ERRORS
    assert summary["errors"][0] == "gpu: TimeoutExpired 0" and summary["errors_dropped"] == 20


def test_watchdog_thread_samples_until_stopped(no_kill):
    sampled = threading.Event()
    sampler = Sequence(watched(100))

    def recording(pid):
        sampled.set()
        return sampler(pid)

    dog = MemoryWatchdog(os.getpid(), interval=0.01, limits=NativeLimits(), on_violation=lambda reason: None,
                         sampler=recording).start()
    assert sampled.wait(5.0)
    summary = dog.stop()
    assert summary["started"] is True and summary["sample_count"] >= 1
    assert not dog._thread.is_alive()
    assert dog.stop() == summary  # idempotent: nothing samples after stop
    # Closed means closed: a direct sample after stop() changes nothing.
    assert dog.sample_once() is None and dog.summary() == summary
    with pytest.raises(RuntimeError, match="started once"):
        dog.start()


def test_a_watchdog_stopped_before_it_started_cannot_be_started(no_kill):
    dog = MemoryWatchdog(42, interval=1.0, limits=NativeLimits(), on_violation=lambda reason: None,
                         sampler=Sequence(watched(100)))
    summary = dog.stop()
    assert summary["started"] is False and summary["sample_count"] == 0 and summary["samples"] == []
    with pytest.raises(RuntimeError, match="never after stop"):
        dog.start()


def test_watchdog_can_be_stopped_from_its_own_violation_callback(no_kill):
    box = {}

    def stop_now(reason):
        box["summary"] = box["dog"].stop()  # joining itself would deadlock; stop() must not

    box["dog"] = MemoryWatchdog(42, interval=0.01, limits=NativeLimits(), on_violation=stop_now,
                                sampler=Sequence(watched(100, pressure=4)))
    box["dog"].start()
    deadline = time.monotonic() + 5.0
    while "summary" not in box and time.monotonic() < deadline:
        time.sleep(0.01)
    final = box["dog"].stop()
    assert box["summary"]["violations"][0]["kind"] == "pressure_critical"
    assert final["sample_count"] == box["summary"]["sample_count"] == 1 and final == box["summary"]


def test_watchdog_stop_is_bounded_when_a_sampler_hangs(no_kill):
    entered, release = threading.Event(), threading.Event()
    reasons = []

    def hung(pid):
        entered.set()
        release.wait(5.0)
        return watched(100, pressure=4)  # a violation that arrives only after the owner moved on to cleanup

    dog = MemoryWatchdog(42, interval=0.01, limits=NativeLimits(), on_violation=reasons.append,
                         sampler=hung, join_timeout=0.05).start()
    try:
        assert entered.wait(5.0)
        summary = dog.stop()
        assert summary["errors"] == ["watchdog thread did not stop within 0.05 s"]
    finally:
        release.set()
    dog._thread.join(5.0)
    assert not dog._thread.is_alive()
    # The late sample is discarded: the summary stop() returned is final and nobody is called back.
    assert reasons == [] and dog.summary() == summary and summary["sample_count"] == 0


def test_watchdog_survives_a_broken_clock(no_kill):
    def clock():
        raise RuntimeError("clock unavailable")

    dog = MemoryWatchdog(42, interval=0.01, limits=NativeLimits(), on_violation=lambda reason: None,
                         sampler=Sequence(watched(100)), clock=clock).start()
    time.sleep(0.05)
    summary = dog.stop()
    # The error is recorded once (deduplicated), the loop kept its interval, and the thread ended on stop().
    assert summary["errors"] == ["watchdog: RuntimeError: clock unavailable"] and not dog._thread.is_alive()


def test_watchdog_rejects_nonsense_configuration():
    limits = NativeLimits()
    for kwargs in ({"interval": 0}, {"interval": float("nan")}, {"interval": 1, "max_samples": 1},
                   {"interval": 1, "budget_bytes": 0}, {"interval": 1, "on_violation": None},
                   {"interval": 1, "join_timeout": float("inf")}, {"interval": 1, "baseline": [1, 2]}):
        arguments = {"limits": limits, "on_violation": lambda reason: None, **kwargs}
        with pytest.raises(ValueError):
            MemoryWatchdog(42, **arguments)


# -------------------------------------------------------------------------------------------- live, darwin only


@DARWIN_ONLY
def test_live_phys_footprint_and_rss_of_this_process():
    footprint, rss = process_phys_footprint(os.getpid()), process_rss(os.getpid())
    assert type(footprint) is int and footprint > 0
    assert type(rss) is int and rss > 0
    assert process_alive(os.getpid()) is True


@DARWIN_ONLY
def test_live_exited_processes_have_no_footprint():
    child = subprocess.Popen(["/usr/bin/true"])
    try:
        deadline = time.monotonic() + 5.0
        while process_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        # A zombie awaiting wait(): libproc answers with footprint 0, which measures nothing.
        assert process_alive(child.pid) is False and process_phys_footprint(child.pid) is None
        assert process_rss(child.pid) is None
    finally:
        child.wait(5)


def test_footprint_is_unavailable_off_darwin(monkeypatch):
    monkeypatch.setattr(apple.sys, "platform", "linux")
    assert process_phys_footprint(os.getpid()) is None
