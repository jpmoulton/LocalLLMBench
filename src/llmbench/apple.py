"""Apple Silicon unified-memory telemetry: labelled evidence, never VRAM. Importing this module probes nothing.

On an Apple Silicon Mac the CPU, the GPU and every other application draw from ONE physical memory pool, so the
NVIDIA notion of "VRAM in use" does not exist and nothing here is ever reported as VRAM. What is measured instead,
each under its own explicit key:

* the server process's `phys_footprint` (libproc `proc_pid_rusage`), which is what Activity Monitor calls Memory
  and, unlike RSS, includes the Metal buffers llama.cpp allocates (they are `VM_ALLOCATE` regions in
  `footprint -p`: 1905 MB for Qwen3-1.7B-Q4_K_M with a 1631 MiB Metal working set on the M1 this was captured on),
  plus its RSS for comparison;
* the host's total/available/used memory (psutil), host-wide swap (`sysctl vm.swapusage`) and swap traffic
  (`vm_stat`), and macOS's own verdict on all of it, `kern.memorystatus_vm_pressure_level` (1 normal, 2 warn,
  4 critical: the libdispatch levels, which is why `NativeLimits.max_memory_pressure_level` is `Literal[1, 2, 4]`);
* the GPU's own utilisation and driver memory (`ioreg` IOAccelerator `PerformanceStatistics`). The GPU is busy even
  at rest (the window server composites on it: 34 % in the captured idle sample), which is why admission compares
  a median across samples to a limit instead of demanding an idle GPU;
* the power source (`pmset -g batt`), because a Mac on battery may be throttled and a result must say so.

Every external probe is one of a few fixed argv lists run with a timeout of at most 3 s in a fixed environment
(system PATH, `LC_ALL=C`: `sysctl vm.swapusage` prints `9216,00M` under a German or French locale), output over
1 MiB is refused rather than parsed, and a failure becomes an entry in `errors` with its field left None: a sample
never raises and never invents a value. Only admission (`check_unified_memory_headroom`) turns missing evidence into a
refusal, because admitting a load onto memory it could not see is exactly the failure this module exists to
prevent. Nothing here stops, signals or renices any process: the watchdog reports a violation to its owner, and the
owner (the native runner) decides what to stop.
"""

from __future__ import annotations

import json
import math
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from functools import lru_cache

MIB = 1024 * 1024
# The ceiling on any single external probe. A caller may ask for less (a phase with little time left), never more.
MAX_PROBE_SECONDS = 3.0
# The IOAccelerator dump is ~34 KB on an M1; anything near this cap is not the output this parser understands.
MAX_COMMAND_OUTPUT_CHARS = 1_048_576
MAX_ERROR_CHARS = 500
MAX_WATCHDOG_ERRORS = 50
# kern.memorystatus_vm_pressure_level reports libdispatch's DISPATCH_MEMORYPRESSURE_* levels. Any other value is
# not a level this module can compare to a limit, so it is refused rather than guessed.
PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}
PRESSURE_CRITICAL = 4
RUSAGE_INFO_V2 = 2
# pid_t is a 32-bit int on darwin; a larger "pid" is a caller bug, not a process libproc could be asked about.
MAX_PID = 2**31 - 1
# Every probe runs in THIS environment, never the caller's: the tools resolve from the system directories only (a
# `sysctl` earlier on a user's PATH is not the one these parsers were written against), and the C locale keeps
# numbers in the shape the parsers expect -- sysctl honours the numeric locale (observed on macOS 14.2.1:
# `vm.swapusage` under de_DE or fr_FR reads `total = 9216,00M`), which would otherwise disable the watchdog's swap
# rule for every such user.
PROBE_ENVIRONMENT = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}

PRESSURE_ARGV = ("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
SWAPUSAGE_ARGV = ("sysctl", "-n", "vm.swapusage")
VM_STAT_ARGV = ("vm_stat",)
IOREG_ACCELERATOR_ARGV = ("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
PMSET_BATT_ARGV = ("pmset", "-g", "batt")
CHIP_ARGV = ("sysctl", "-n", "machdep.cpu.brand_string")
MEMSIZE_ARGV = ("sysctl", "-n", "hw.memsize")
PERFORMANCE_CORES_ARGV = ("sysctl", "-n", "hw.perflevel0.physicalcpu")
EFFICIENCY_CORES_ARGV = ("sysctl", "-n", "hw.perflevel1.physicalcpu")

# ioreg prints one `+-o <class ...>` header per matched IOAccelerator and its properties as `"key" = value` lines;
# PerformanceStatistics is a flat dict of `"name"=integer` pairs. Keys are matched EXACTLY: "In use system memory
# (driver)" sits next to "In use system memory" and is a different number once a model is resident.
_IOREG_ENTRY = re.compile(r"^(?=\+-o )", re.MULTILINE)
_IOREG_STATS = re.compile(r'^[ \t|]*"PerformanceStatistics" = \{([^{}\n]*)\}[ \t]*$', re.MULTILINE)
_IOREG_STAT_PAIR = re.compile(r'"([^"=\n]+)"=(-?\d+)')
_IOREG_CORES = re.compile(r'^[ \t|]*"gpu-core-count" = (\d+)[ \t]*$', re.MULTILINE)
_IOREG_MODEL = re.compile(r'^[ \t|]*"model" = "([^"\n]*)"[ \t]*$', re.MULTILINE)
_IOREG_FIELDS = (("device_utilization_percent", "Device Utilization %"),
                 ("renderer_utilization_percent", "Renderer Utilization %"),
                 ("tiler_utilization_percent", "Tiler Utilization %"),
                 ("in_use_system_memory_bytes", "In use system memory"),
                 ("alloc_system_memory_bytes", "Alloc system memory"))

_POWER_SOURCE = re.compile(r"^Now drawing from '([^'\n]+)'[ \t]*$", re.MULTILINE)
_BATTERY_LINE = re.compile(r"^[ \t]*-InternalBattery-\d+\b[^\n]*?\s(\d{1,3})%;[ \t]*([^;\n]+?)[ \t]*;", re.MULTILINE)
_POWER_SOURCES = {"AC Power": "ac", "Battery Power": "battery"}
# pmset's battery states. "AC attached; not charging" (optimised charging, or a full battery held on AC) is not
# charging; an unlisted state is recorded as its label with `charging` None rather than mapped by resemblance.
_CHARGING_STATES = {"charging": True, "finishing charge": True, "discharging": False, "charged": False,
                    "ac attached": False}

_VM_STAT_HEADER = re.compile(r"^Mach Virtual Memory Statistics: \(page size of (\d+) bytes\)[ \t]*$")
_VM_STAT_LINE = re.compile(r'^"?([A-Za-z][A-Za-z -]*?)"?:[ \t]+(\d+)\.?[ \t]*$')
_VM_STAT_REQUIRED = ("page_size_bytes", "pages_free", "pageouts", "swapins", "swapouts")

_SWAP_FIELD = re.compile(r"\b(total|used|free) = (\d+(?:\.\d+)?)([KMG])(?=\s|$)")
_SWAP_UNITS = {"K": 1024, "M": MIB, "G": 1024 * MIB}


# --------------------------------------------------------------------------------------------------- parsers


def parse_ioreg_accelerator(text: str) -> dict:
    """The one IOAccelerator's utilisation, driver memory, core count and model from `ioreg -r -d 1 -w 0 -c
    IOAccelerator`. A key the driver did not print is None. Raises ValueError unless EXACTLY one accelerator
    reported PerformanceStatistics: every Apple Silicon Mac has one GPU, and a second one is a machine this module
    has never seen, whose numbers it will not attribute to "the" GPU."""
    entries = [entry for entry in _IOREG_ENTRY.split(text) if _IOREG_STATS.search(entry)]
    if len(entries) != 1:
        raise ValueError(f"{len(entries)} IOAccelerator entries reported PerformanceStatistics; exactly one is "
                         "expected on Apple Silicon")
    entry = entries[0]
    statistics_body = _IOREG_STATS.search(entry).group(1)
    values: dict[str, int] = {}
    for key, value in _IOREG_STAT_PAIR.findall(statistics_body):
        if key in values:
            raise ValueError(f"IOAccelerator PerformanceStatistics repeats {key!r}")
        values[key] = int(value)
    row = {name: values.get(key) for name, key in _IOREG_FIELDS}
    for name, value in row.items():
        # A percentage outside 0-100 or a negative byte count is not a reading: a "-1 %" GPU would look idle to
        # admission, so it is refused here instead of compared to a limit.
        ceiling = 100 if name.endswith("_percent") else None
        if value is not None and (value < 0 or (ceiling is not None and value > ceiling)):
            raise ValueError(f"IOAccelerator {name} = {value} is out of range")
    cores, model = _IOREG_CORES.findall(entry), _IOREG_MODEL.findall(entry)
    if len(cores) > 1 or len(model) > 1:
        raise ValueError("IOAccelerator entry repeats gpu-core-count or model")
    row["gpu_core_count"] = int(cores[0]) if cores else None
    row["model"] = model[0] if model else None
    return row


def parse_pmset_batt(text: str) -> dict:
    """Power source and internal battery state from `pmset -g batt`.

    `power_source` is "ac" or "battery" only for those two labels; a UPS or any other source is None with its label
    kept in `power_source_label`. A Mac without an internal battery (a Mac mini, a Studio) prints no battery line,
    so `battery_percent` and `charging` are None: absent, not zero and not "charged"."""
    sources = _POWER_SOURCE.findall(text)
    if len(sources) != 1:
        raise ValueError("pmset output does not name exactly one power source")
    lines = [line for line in text.splitlines() if "-InternalBattery-" in line]
    if len(lines) > 1:
        raise ValueError("pmset reports more than one internal battery")
    percent = state = None
    if lines:
        # A battery line in a shape this parser does not know is an error, not an absent battery.
        match = _BATTERY_LINE.match(lines[0])
        if match is None:
            raise ValueError(f"unrecognised pmset battery line: {lines[0].strip()[:80]!r}")
        percent, state = int(match.group(1)), match.group(2)
        if percent > 100:
            raise ValueError(f"pmset battery percentage {percent} is out of range")
    return {"power_source": _POWER_SOURCES.get(sources[0]), "power_source_label": sources[0],
            "battery_percent": percent, "battery_state": state,
            "charging": None if state is None else _CHARGING_STATES.get(state.lower())}


def parse_vm_stat(text: str) -> dict:
    """Every counter `vm_stat` prints, as ints under snake_case keys (`Pages free` -> `pages_free`,
    `"Translation faults"` -> `translation_faults`), plus `page_size_bytes` from its header.

    Counters are pages or events since boot; `swapins`/`swapouts` are cumulative, so the watchdog reports their
    DELTA over a candidate. A line in any other shape is a format this parser does not know and raises ValueError
    rather than being skipped, so a changed `vm_stat` shows up as a recorded error, not a silently missing key."""
    lines = [line for line in text.splitlines() if line.strip()]
    header = _VM_STAT_HEADER.match(lines[0]) if lines else None
    if header is None:
        raise ValueError("vm_stat output has no page-size header")
    row = {"page_size_bytes": int(header.group(1))}
    for line in lines[1:]:
        match = _VM_STAT_LINE.match(line)
        if match is None:
            raise ValueError(f"unrecognised vm_stat line: {line.strip()[:80]!r}")
        key = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
        if key in row:
            raise ValueError(f"vm_stat repeats {key}")
        row[key] = int(match.group(2))
    missing = [key for key in _VM_STAT_REQUIRED if key not in row]
    if missing:
        raise ValueError("vm_stat output lacks " + ", ".join(missing))
    return row


def parse_swapusage(text: str) -> dict:
    """Host-wide swap from `sysctl -n vm.swapusage` (`total = 9216.00M  used = 7798.56M  free = 1417.44M
    (encrypted)`), in bytes. sysctl prints MiB to two decimals, so each value is exact to about 10 KiB; that is
    the resolution of the source, not of this conversion."""
    fields: dict[str, int] = {}
    for name, number, unit in _SWAP_FIELD.findall(text):
        if name in fields:
            raise ValueError(f"vm.swapusage repeats {name}")
        fields[name] = round(float(number) * _SWAP_UNITS[unit])
    missing = [name for name in ("total", "used", "free") if name not in fields]
    if missing:
        raise ValueError("vm.swapusage output lacks " + ", ".join(missing))
    return {"swap_total_bytes": fields["total"], "swap_used_bytes": fields["used"],
            "swap_free_bytes": fields["free"], "swap_encrypted": "(encrypted)" in text}


def parse_pressure_level(text: str) -> int:
    """`sysctl -n kern.memorystatus_vm_pressure_level`: 1 normal, 2 warn or 4 critical, nothing else."""
    value = text.strip()
    if not value.isdigit() or int(value) not in PRESSURE_LEVELS:
        raise ValueError(f"unrecognised memory pressure level {value[:20]!r}; expected 1, 2 or 4")
    return int(value)


def _parse_sysctl_int(text: str) -> int:
    value = text.strip()
    if not value.isdigit():
        raise ValueError(f"sysctl printed {value[:40]!r}, not a non-negative integer")
    return int(value)


def _parse_sysctl_text(text: str) -> str:
    value = text.strip()
    if not value or "\n" in value:
        raise ValueError("sysctl printed no single-line value")
    return value


# ---------------------------------------------------------------------------------------------- probe plumbing


def _probe_timeout(timeout) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) \
            or timeout <= 0:
        raise ValueError("probe timeout must be a finite positive number of seconds")
    return min(float(timeout), MAX_PROBE_SECONDS)


def _check_pid(pid) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid <= MAX_PID:
        raise ValueError(f"pid must be a positive integer no larger than {MAX_PID}")
    return pid


def _run_probe(command_runner, argv: tuple[str, ...], timeout: float) -> str:
    """One fixed argv in PROBE_ENVIRONMENT, bounded in time by `timeout` and in size by MAX_COMMAND_OUTPUT_CHARS.
    A nonzero exit is a failure even from a runner that ignored `check=True`."""
    completed = command_runner(list(argv), capture_output=True, text=True, timeout=timeout, check=True,
                               env=dict(PROBE_ENVIRONMENT))
    returncode = getattr(completed, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):  # no exit status is not a success
        raise ValueError("probe runner reported no integer exit status")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(argv))
    output = getattr(completed, "stdout", None)
    if not isinstance(output, str):
        raise ValueError("probe returned no text output")
    if len(output) > MAX_COMMAND_OUTPUT_CHARS:
        raise ValueError(f"probe output exceeds {MAX_COMMAND_OUTPUT_CHARS} characters")
    return output


def _error(label: str, exc: BaseException) -> str:
    return f"{label}: {type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


def _attempt(errors: list[str], label: str, action):
    """Run one probe or reader; a failure is recorded under `label` and yields None, never an exception.
    RuntimeError is included because it is how psutil reports a failed `host_statistics64` on macOS."""
    try:
        return action()
    except (OSError, subprocess.SubprocessError, ValueError, ImportError, RuntimeError) as exc:
        errors.append(_error(label, exc))
        return None


def _int_or_none(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ---------------------------------------------------------------------------------------------- process memory


@lru_cache(maxsize=1)
def _libproc():
    """libproc's `proc_pid_rusage`, loaded on first use and never at import. The struct is `rusage_info_v2` from
    <sys/resource.h>; only `ri_phys_footprint` and `ri_proc_exit_abstime` are read."""
    import ctypes

    class RusageInfoV2(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
            "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten")]

    function = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True).proc_pid_rusage
    function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    function.restype = ctypes.c_int
    return ctypes, function, RusageInfoV2


def process_phys_footprint(pid: int) -> int | None:
    """The process's `phys_footprint` in bytes (Activity Monitor's Memory, including Metal allocations), or None
    when it cannot be read: not a darwin host, no such process (ESRCH), not ours to read (EPERM), or a process that
    has already exited and awaits its parent's wait, for which libproc reports 0 -- a zero that measures nothing.

    The pid must be one this caller owns and has not reaped: after a reap the number can be reused by an unrelated
    process, and its footprint would be read as the server's."""
    _check_pid(pid)
    if sys.platform != "darwin":
        return None
    try:
        ctypes, function, info_type = _libproc()
    except (OSError, AttributeError):
        return None
    info = info_type()
    if function(pid, RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        return None
    if info.ri_proc_exit_abstime:
        return None
    return int(info.ri_phys_footprint)


def process_rss(pid: int) -> int | None:
    """Resident set size via psutil, or None (no psutil, no such process, a zombie, access denied). RSS excludes
    most of what Metal maps for the GPU, so it is recorded beside phys_footprint, never instead of it."""
    _check_pid(pid)
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError):
        return None


def process_alive(pid: int) -> bool | None:
    """True while the process runs, False once it is gone or a zombie, None when psutil cannot say."""
    _check_pid(pid)
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:  # includes ZombieProcess
        return False
    except (psutil.Error, OSError):
        return None


def _host_memory(virtual_memory) -> dict:
    if virtual_memory is None:
        import psutil
        virtual_memory = psutil.virtual_memory
    memory = virtual_memory()
    values = {"host_memory_total_bytes": memory.total, "host_memory_available_bytes": memory.available,
              "host_memory_used_bytes": memory.used}
    for key, value in values.items():
        if _int_or_none(value) is None or value < 0:
            raise ValueError(f"{key} is not a non-negative integer")
    return values


# ------------------------------------------------------------------------------------------------------ samples


def sample_unified_memory(pid: int | None = None, *, command_runner=subprocess.run,
                          timeout: float = MAX_PROBE_SECONDS, include_gpu: bool = True, include_power: bool = True,
                          footprint_reader=None, rss_reader=None, alive_reader=None, virtual_memory=None,
                          clock=time.monotonic) -> dict:
    """One labelled unified-memory sample (`kind` "apple-unified"). Never raises for a probe failure: the field
    stays None and the failure is an entry in `errors`.

    `timeout` bounds EACH external command and is clamped to MAX_PROBE_SECONDS; with every probe enabled a sample
    runs five commands, typically in ~50 ms on an M1. `pid`, when given, adds `process` (the server's
    phys_footprint, RSS and whether it is alive). `include_gpu`/`include_power` False leave `gpu`/`power` None
    WITHOUT running ioreg/pmset -- not attempted, so not an error. The readers default to this module's own
    (`process_phys_footprint`, `process_rss`, `process_alive`, psutil's `virtual_memory`) and are injectable for
    tests; `sample_seconds` is how long the sample itself took, which is the watchdog's own latency.
    """
    probe_timeout = _probe_timeout(timeout)
    if pid is not None:
        _check_pid(pid)
    errors: list[str] = []
    started = clock()
    row: dict = {"kind": "apple-unified", "monotonic_seconds": started, "host_memory_total_bytes": None,
                 "host_memory_available_bytes": None, "host_memory_used_bytes": None, "swap_total_bytes": None,
                 "swap_used_bytes": None, "swapins": None, "swapouts": None, "memory_pressure_level": None,
                 "process": None, "gpu": None, "power": None, "errors": errors}
    row.update(_attempt(errors, "host memory", lambda: _host_memory(virtual_memory)) or {})
    row["memory_pressure_level"] = _attempt(errors, "memory pressure", lambda: parse_pressure_level(
        _run_probe(command_runner, PRESSURE_ARGV, probe_timeout)))
    swap = _attempt(errors, "swap usage", lambda: parse_swapusage(
        _run_probe(command_runner, SWAPUSAGE_ARGV, probe_timeout)))
    if swap is not None:
        row["swap_total_bytes"], row["swap_used_bytes"] = swap["swap_total_bytes"], swap["swap_used_bytes"]
    vm = _attempt(errors, "vm_stat", lambda: parse_vm_stat(_run_probe(command_runner, VM_STAT_ARGV, probe_timeout)))
    if vm is not None:
        row["swapins"], row["swapouts"] = vm["swapins"], vm["swapouts"]
    if pid is not None:
        readers = (("alive", alive_reader or process_alive), ("phys_footprint_bytes",
                   footprint_reader or process_phys_footprint), ("rss_bytes", rss_reader or process_rss))
        process: dict = {"pid": pid}
        for key, reader in readers:
            process[key] = _attempt(errors, f"process {key}", lambda reader=reader: reader(pid))
        if process["alive"] is True and process["phys_footprint_bytes"] is None \
                and not any(error.startswith("process phys_footprint_bytes:") for error in errors):
            # The reader answers None rather than raising; for a process that is still running that None is a
            # failed measurement (and a footprint rule the watchdog cannot apply), so it is said, not implied.
            errors.append("process phys_footprint_bytes: unreadable for a running process")
        row["process"] = process
    if include_gpu:
        row["gpu"] = _attempt(errors, "gpu", lambda: parse_ioreg_accelerator(
            _run_probe(command_runner, IOREG_ACCELERATOR_ARGV, probe_timeout)))
    if include_power:
        row["power"] = _attempt(errors, "power", lambda: parse_pmset_batt(
            _run_probe(command_runner, PMSET_BATT_ARGV, probe_timeout)))
    row["sample_seconds"] = max(0.0, clock() - started)
    json.dumps(row, allow_nan=False)
    return row


def host_facts(*, command_runner=subprocess.run, timeout: float = MAX_PROBE_SECONDS) -> dict:
    """STABLE facts about this Mac for a bundle's provenance: architecture, chip, installed memory, macOS version,
    performance/efficiency core counts and GPU core count. Free memory, swap and pressure change from one second
    to the next and are deliberately NOT here; they belong to samples. A fact that could not be read is None with
    the reason in `errors` (an Intel Mac has no `hw.perflevel1`, for one)."""
    probe_timeout = _probe_timeout(timeout)
    errors: list[str] = []

    def sysctl(label, argv, parse):
        return _attempt(errors, label, lambda: parse(_run_probe(command_runner, argv, probe_timeout)))

    gpu = _attempt(errors, "gpu", lambda: parse_ioreg_accelerator(
        _run_probe(command_runner, IOREG_ACCELERATOR_ARGV, probe_timeout)))
    return {"machine": platform.machine() or None,
            "chip": sysctl("chip", CHIP_ARGV, _parse_sysctl_text),
            "memory_bytes": sysctl("memory", MEMSIZE_ARGV, _parse_sysctl_int),
            "macos": platform.mac_ver()[0] or None,
            "performance_cores": sysctl("performance cores", PERFORMANCE_CORES_ARGV, _parse_sysctl_int),
            "efficiency_cores": sysctl("efficiency cores", EFFICIENCY_CORES_ARGV, _parse_sysctl_int),
            "gpu_core_count": None if gpu is None else gpu["gpu_core_count"],
            "errors": errors}


# ---------------------------------------------------------------------------------------------------- admission


def _usable(sample) -> bool:
    return (isinstance(sample, dict) and sample.get("kind") == "apple-unified"
            and _int_or_none(sample.get("host_memory_available_bytes")) is not None
            and _int_or_none(sample.get("memory_pressure_level")) is not None)


def check_unified_memory_headroom(samples: list[dict], *, required_bytes: int, limits) -> dict:
    """Admit a native load only when the samples SHOW room for it. A competing workload is a conflict to report,
    never a process to stop.

    `required_bytes` is what the caller needs available (the model file plus `limits.memory_reserve_mib`). The
    worst reading wins for memory -- the LOWEST available bytes and the HIGHEST pressure level across usable
    samples -- while GPU utilisation is judged on its median, because the window server's own compositing makes
    single readings spiky. Missing evidence refuses: no usable sample, or no GPU utilisation in any of them, is a
    PreflightError just like `check_gpu_headroom`'s unavailable telemetry. Every failed condition is reported
    together. Returns the evidence dict on admission.
    """
    from .containers.preflight import PreflightError

    if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes < 0:
        raise ValueError("required_bytes must be a non-negative integer")
    usable = [sample for sample in samples or () if _usable(sample)]
    if not usable:
        errors = sorted({error for sample in samples or () if isinstance(sample, dict)
                         for error in sample.get("errors") or ()})
        raise PreflightError("unified-memory telemetry is unavailable: no sample reported both available memory "
                             "and the memory pressure level" + (" (" + "; ".join(errors)[:1000] + ")" if errors
                                                                else ""))
    available = min(sample["host_memory_available_bytes"] for sample in usable)
    pressure = max(sample["memory_pressure_level"] for sample in usable)
    gpu_samples = [value for sample in usable
                   if (value := _int_or_none((sample.get("gpu") or {}).get("device_utilization_percent")))
                   is not None]
    gpu_limit = limits.max_foreign_gpu_utilization_percent
    problems = []
    if available < required_bytes:
        problems.append(f"the host has {available / MIB:.0f} MiB of unified memory available; this candidate "
                        f"needs {required_bytes / MIB:.0f} MiB (model plus reserve)")
    if pressure > limits.max_memory_pressure_level:
        problems.append(f"macOS memory pressure is level {pressure} ({PRESSURE_LEVELS.get(pressure, 'unknown')}); "
                        f"the limit for admission is {limits.max_memory_pressure_level}")
    median = statistics.median(gpu_samples) if gpu_samples else None
    if median is None:
        problems.append("GPU utilization telemetry is unavailable (no IOAccelerator Device Utilization reading)")
    elif median > gpu_limit:
        problems.append(f"GPU device utilization is already {median:g}% (median of {len(gpu_samples)} samples); "
                        f"max_foreign_gpu_utilization_percent is {gpu_limit}")
    if problems:
        raise PreflightError("; ".join(problems) + ". A competing workload is a conflict to report, never a "
                             "process to stop")
    latest = usable[-1]
    return {"kind": "apple-unified", "available_bytes": available, "required_bytes": required_bytes,
            "pressure_level": pressure, "max_pressure_level": limits.max_memory_pressure_level,
            "gpu_utilization_percent_samples": gpu_samples, "gpu_utilization_percent_median": median,
            "max_foreign_gpu_utilization_percent": gpu_limit, "swap_used_bytes": latest.get("swap_used_bytes"),
            "power": latest.get("power"), "samples": len(usable), "verdict": "admitted"}


# ----------------------------------------------------------------------------------------------------- watchdog


class MemoryWatchdog:
    """Samples unified memory while a native server runs and REPORTS the first limit it sees broken.

    There is no cgroup around a macOS host process, so this is the only thing between a candidate and a Mac that
    swaps itself to a standstill. It watches three rules, each against `limits` (a `NativeLimits`):

    * `footprint`: the server's phys_footprint above `limits.max_server_footprint_mib`, or, when that is None,
      above `budget_bytes` -- Metal's recommended working set as the server logged it. With neither there is no
      footprint rule, and the summary says so (`footprint_limit_source` "none");
    * `swap_growth`: host swap in use more than `limits.max_swap_growth_mib` above its starting value, which is
      `baseline`'s (an admission sample taken before the load) when given, else the first sample's;
    * `pressure_critical`: macOS memory pressure at level 4.

    `on_violation(reason)` is called ONCE, for the first violation, from the sampling thread; later violations are
    still recorded in the summary. The watchdog itself never signals, kills or renices anything. `sampler(pid)`
    must return a `sample_unified_memory`-shaped dict; its exceptions, a non-dict, and an exception from
    `on_violation` become summary `errors`, never an exception in the thread. Retained samples are bounded by
    `max_samples` (oldest dropped), while counts, peaks, minima and deltas are kept over EVERY sample; the GPU
    utilisation median alone is over the retained window, and says how many readings it covers.

    Once `stop()` has returned, the watchdog is closed: its summary is final. A sampler that outlived the bounded
    join and returns later is discarded -- it neither changes that summary nor calls `on_violation` into an owner
    that has already moved on to cleanup.
    """

    def __init__(self, pid: int, *, interval: float, limits, on_violation, budget_bytes: int | None = None,
                 sampler=sample_unified_memory, clock=time.monotonic, max_samples: int = 20000,
                 baseline: dict | None = None, join_timeout: float = 20.0) -> None:
        _check_pid(pid)
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) \
                or interval <= 0:
            raise ValueError("watchdog interval must be a finite positive number of seconds")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer of at least 2")
        if budget_bytes is not None and (_int_or_none(budget_bytes) is None or budget_bytes <= 0):
            raise ValueError("budget_bytes must be a positive integer or None")
        if not callable(on_violation) or not callable(sampler):
            raise ValueError("on_violation and sampler must be callable")
        if isinstance(join_timeout, bool) or not isinstance(join_timeout, (int, float)) \
                or not math.isfinite(join_timeout) or join_timeout <= 0:
            raise ValueError("join_timeout must be a finite positive number of seconds")
        if baseline is not None and not isinstance(baseline, dict):
            raise ValueError("baseline must be a sample dict or None")
        self.pid, self.interval, self.limits = pid, float(interval), limits
        self.sampler, self.on_violation, self.clock = sampler, on_violation, clock
        self.max_samples, self.join_timeout = max_samples, float(join_timeout)
        if limits.max_server_footprint_mib is not None:
            self.footprint_limit_bytes = limits.max_server_footprint_mib * MIB
            self.footprint_limit_source = "max_server_footprint_mib"
        elif budget_bytes is not None:
            self.footprint_limit_bytes, self.footprint_limit_source = budget_bytes, "metal_budget"
        else:
            self.footprint_limit_bytes, self.footprint_limit_source = None, "none"
        self.swap_growth_limit_bytes = limits.max_swap_growth_mib * MIB
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: deque = deque(maxlen=max_samples)
        self._gpu: deque = deque(maxlen=max_samples)
        self._gpu_max = None
        self._gpu_count = 0
        self._closed = False
        self._count = 0
        self._first = self._last = None
        self._peak_footprint = self._peak_rss = self._min_available = self._max_pressure = None
        self._swap_start = self._swap_peak = self._swap_end = None
        self._swap_baseline = None
        self._swapins = [None, None]
        self._swapouts = [None, None]
        self._violations: list[dict] = []
        self._notified = False
        self._errors: list[str] = []
        self._errors_dropped = 0
        self._samples_with_errors = 0
        if baseline is not None:
            swap = _int_or_none(baseline.get("swap_used_bytes"))
            if swap is not None:
                self._swap_start = self._swap_peak = swap
                self._swap_baseline = "baseline"
            self._swapins[0] = _int_or_none(baseline.get("swapins"))
            self._swapouts[0] = _int_or_none(baseline.get("swapouts"))

    # -- lifecycle ---------------------------------------------------------------------------------------------

    def start(self) -> "MemoryWatchdog":
        with self._lock:
            if self._thread is not None or self._closed:
                raise RuntimeError("a memory watchdog can be started once, and never after stop()")
            self._thread = threading.Thread(target=self._run, name="llmbench-memory-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        """Stop sampling, close the watchdog and return its final summary. Idempotent, and safe to call from
        `on_violation` (the sampling thread is then not joined, it simply ends after the current sample). Bounded
        by `join_timeout`; a sampler that outlives it is reported in `errors`, never waited on forever, and
        whatever it returns afterwards is discarded."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.join_timeout)
            if thread.is_alive():
                with self._lock:
                    self._record_error(f"watchdog thread did not stop within {self.join_timeout:g} s")
        with self._lock:
            self._closed = True
        return self.summary()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                began = self.clock()
                self.sample_once()
                delay = max(0.0, self.interval - (self.clock() - began))
            except Exception as exc:  # the thread never raises: a broken clock is an error, not a dead watchdog
                with self._lock:
                    self._record_error(_error("watchdog", exc))
                delay = self.interval
            if self._stop.wait(delay):
                break

    # -- sampling ----------------------------------------------------------------------------------------------

    def sample_once(self) -> dict | None:
        """Take, record and judge one sample; returns it (None when the sampler failed, or when the watchdog was
        closed by `stop()` before the sample came back). The thread calls this every `interval`; a caller may
        also call it directly, which is how the tests drive it deterministically."""
        try:
            sample = self.sampler(self.pid)
        except Exception as exc:
            with self._lock:
                self._record_error(_error("sampler", exc))
            return None
        if not isinstance(sample, dict):
            with self._lock:
                self._record_error(f"sampler returned {type(sample).__name__}, not a sample")
            return None
        with self._lock:
            if self._closed:
                return None
            self._record(sample)
            fresh = self._judge(sample)
            reason = None
            if fresh and not self._notified:
                self._notified, reason = True, fresh[0]["reason"]
        if reason is not None:
            try:
                self.on_violation(reason)
            except Exception as exc:
                with self._lock:
                    self._record_error(_error("on_violation", exc))
        return sample

    def _record_error(self, message: str) -> None:
        if self._closed or message in self._errors:  # a closed watchdog's summary is final
            return
        if len(self._errors) >= MAX_WATCHDOG_ERRORS:
            self._errors_dropped += 1
            return
        self._errors.append(message)

    def _record(self, sample: dict) -> None:
        self._count += 1
        self._samples.append(sample)
        self._first = self._first if self._first is not None else sample
        self._last = sample
        errors = sample.get("errors")
        if errors:
            self._samples_with_errors += 1
            for error in errors if isinstance(errors, (list, tuple)) else [errors]:
                self._record_error(str(error)[:MAX_ERROR_CHARS])
        process = sample.get("process") if isinstance(sample.get("process"), dict) else {}
        self._peak_footprint = _peak(self._peak_footprint, _int_or_none(process.get("phys_footprint_bytes")), max)
        self._peak_rss = _peak(self._peak_rss, _int_or_none(process.get("rss_bytes")), max)
        self._min_available = _peak(self._min_available, _int_or_none(sample.get("host_memory_available_bytes")),
                                    min)
        self._max_pressure = _peak(self._max_pressure, _int_or_none(sample.get("memory_pressure_level")), max)
        swap = _int_or_none(sample.get("swap_used_bytes"))
        if swap is not None:
            if self._swap_start is None:
                self._swap_start, self._swap_baseline = swap, "first-sample"
            self._swap_peak = _peak(self._swap_peak, swap, max)
            self._swap_end = swap
        for counter, key in ((self._swapins, "swapins"), (self._swapouts, "swapouts")):
            value = _int_or_none(sample.get(key))
            if value is not None:
                counter[0] = value if counter[0] is None else counter[0]
                counter[1] = value
        gpu = sample.get("gpu") if isinstance(sample.get("gpu"), dict) else {}
        utilization = _int_or_none(gpu.get("device_utilization_percent"))
        if utilization is not None:
            self._gpu.append(utilization)
            self._gpu_count += 1
            self._gpu_max = _peak(self._gpu_max, utilization, max)

    def _judge(self, sample: dict) -> list[dict]:
        """The violations this sample shows for the FIRST time, one per rule kind."""
        found = []
        process = sample.get("process") if isinstance(sample.get("process"), dict) else {}
        footprint = _int_or_none(process.get("phys_footprint_bytes"))
        if self.footprint_limit_bytes is not None and footprint is not None and footprint > self.footprint_limit_bytes:
            found.append(("footprint", f"server phys_footprint {footprint / MIB:.0f} MiB exceeds the "
                          f"{self.footprint_limit_bytes / MIB:.0f} MiB limit ({self.footprint_limit_source})"))
        swap = _int_or_none(sample.get("swap_used_bytes"))
        if swap is not None and self._swap_start is not None and swap - self._swap_start > self.swap_growth_limit_bytes:
            found.append(("swap_growth", f"host swap in use grew by {(swap - self._swap_start) / MIB:.0f} MiB; "
                          f"the limit is {self.limits.max_swap_growth_mib} MiB (swap is host-wide)"))
        if _int_or_none(sample.get("memory_pressure_level")) == PRESSURE_CRITICAL:
            found.append(("pressure_critical", "macOS memory pressure reached level 4 (critical)"))
        seen = {item["kind"] for item in self._violations}
        fresh = [{"kind": kind, "reason": reason, "monotonic_seconds": sample.get("monotonic_seconds")}
                 for kind, reason in found if kind not in seen]
        self._violations.extend(fresh)
        return fresh

    # -- summary -----------------------------------------------------------------------------------------------

    def samples(self) -> list[dict]:
        """The retained samples (at most `max_samples`, newest last), for the run directory."""
        with self._lock:
            return list(self._samples)

    def summary(self) -> dict:
        """The evidence so far (final once `stop()` returned). `samples` is the retained sample LIST, newest last
        -- large, so an owner persists it to its own file and keeps the rest -- and `sample_count` is how many
        samples were taken in all; every other key is a count, extreme, delta or setting."""
        with self._lock:
            gpu = list(self._gpu)
            # Growth is the largest rise over the start that a SAMPLE observed: a baseline alone measures nothing.
            growth = (None if self._swap_start is None or self._swap_end is None
                      else self._swap_peak - self._swap_start)
            return {"kind": "apple-unified-watchdog", "pid": self.pid, "interval_seconds": self.interval,
                    "started": self._thread is not None, "samples": list(self._samples),
                    "sample_count": self._count, "samples_retained": len(self._samples),
                    "samples_with_errors": self._samples_with_errors,
                    "first": self._first, "last": self._last,
                    "peak_phys_footprint_bytes": self._peak_footprint, "peak_rss_bytes": self._peak_rss,
                    "min_available_bytes": self._min_available, "max_pressure_level": self._max_pressure,
                    "swap_baseline": self._swap_baseline, "swap_used_start_bytes": self._swap_start,
                    "swap_used_peak_bytes": self._swap_peak, "swap_used_end_bytes": self._swap_end,
                    "swap_growth_bytes": growth, "swapins_delta": _delta(self._swapins),
                    "swapouts_delta": _delta(self._swapouts),
                    "gpu_utilization_percent": {"median": statistics.median(gpu) if gpu else None,
                                                "median_of_last": len(gpu), "max": self._gpu_max,
                                                "samples": self._gpu_count},
                    "footprint_limit_bytes": self.footprint_limit_bytes,
                    "footprint_limit_source": self.footprint_limit_source,
                    "swap_growth_limit_bytes": self.swap_growth_limit_bytes,
                    "violations": [dict(item) for item in self._violations],
                    "errors": list(self._errors), "errors_dropped": self._errors_dropped}


def _peak(current, value, pick):
    if value is None:
        return current
    return value if current is None else pick(current, value)


def _delta(pair: list) -> int | None:
    return None if pair[0] is None or pair[1] is None else pair[1] - pair[0]
