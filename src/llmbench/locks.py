"""Describing a held lock file. Nothing here ever breaks a lock: that stays a human decision.

An exclusive lock file left behind by a killed process blocks every later run. The old failure was a bare
``FileExistsError`` naming a temp path; the person reading it needs to know who holds the lock and whether that
process still exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def process_alive(pid: int) -> bool | None:
    """Whether a process with this pid exists. ``None`` when it cannot be determined. Never signals the process.

    On Windows ``os.kill(pid, 0)`` is NOT a probe - any signal other than the two console events TERMINATES the
    target - so the process is opened for a limited query instead.
    """
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return True if ctypes.get_last_error() == 5 else False  # access denied: it exists; else: it does not
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                    return None
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(ctypes.c_void_p(handle))
        except Exception:
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _native_holder(holder: dict) -> tuple[bool, int | None]:
    """Whether the holder is a metal-native run, and the llama-server pid it recorded (None when it recorded none).

    A native run's leftover is a host process, not a container, so pointing at ``docker ps`` would send the reader
    to the wrong place. The holder says so itself: ``runtime: "metal-native"``, or the ``native-run:`` owner the
    native runner's lease is written with (which also records ``server_pid`` once the server has started).
    """
    owner = holder.get("owner")
    native = holder.get("runtime") == "metal-native" or (isinstance(owner, str) and owner.startswith("native-run:"))
    server = holder.get("server_pid")
    return native, server if native and type(server) is int and server > 0 else None


def _container_holder(holder: dict) -> bool:
    """Whether the holder says it is a container run: the container runner's lease owner (``container-run`` or
    ``container-run:<attempt>``, the shape every lease had before runtimes existed) or ``runtime:
    "nvidia-container"``. Those texts are unchanged."""
    owner = holder.get("owner")
    return holder.get("runtime") == "nvidia-container" or (
        isinstance(owner, str) and (owner == "container-run" or owner.startswith("container-run:")))


def _native_leftover(server_pid: int | None) -> str:
    """What to check before deleting a stale native lock: the llama-server it started, by pid when recorded."""
    if server_pid is None:
        return ("Check that no llama-server it started is still running (`pgrep -fl llama-server`), then delete "
                "the file")
    alive = process_alive(server_pid)
    if alive is False:
        return (f"Its llama-server, pid {server_pid}, is gone too (`ps -p {server_pid}` confirms it); delete the "
                "file")
    if alive:
        return (f"Its llama-server pid {server_pid} still exists (a reused pid can belong to an unrelated process): "
                f"check it with `ps -p {server_pid}` and stop it only if it is that llama-server, then delete the file")
    return f"Check that its llama-server, pid {server_pid}, is gone (`ps -p {server_pid}`), then delete the file"


MAX_HOLDER_BYTES = 4096


def _read_holder(target: Path) -> bytes:
    """The first ``MAX_HOLDER_BYTES`` of a lock file, and never more.

    A lock lives in the shared temp directory and ``doctor`` describes it whenever it exists, so whatever sits at
    that path is read bounded and non-blocking: a FIFO reads as empty instead of hanging, and a link to
    ``/dev/zero`` or a huge file costs one bounded read, not all of it."""
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    try:
        return os.read(descriptor, MAX_HOLDER_BYTES)
    finally:
        os.close(descriptor)


def describe_lock(path: str | Path) -> str:
    """One sentence for an error message: the holder, whether it is still running, and what to do about it."""
    target = Path(path)
    try:
        holder = json.loads(_read_holder(target).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        holder = None
    if not isinstance(holder, dict) or type(holder.get("pid")) is not int:
        return (f"{target} exists but does not name its holder; confirm no llmbench run is active before "
                "deleting it")
    pid, alive = holder["pid"], process_alive(holder["pid"])
    who = f"pid {pid}" + (f" ({holder['owner']})" if isinstance(holder.get("owner"), str) else "")
    since = f", created {holder['created']}" if isinstance(holder.get("created"), str) else ""
    native, server_pid = _native_holder(holder)
    if alive is False:
        if native:
            leftover = _native_leftover(server_pid)
        elif _container_holder(holder):
            leftover = ("Check that no container was left behind (`docker ps -a --filter name=llmbench-`), then "
                        "delete the file")
        else:
            # A holder that names no runtime -- the campaign lock `tune`, `resume` and the sweeps take records only
            # its pid -- may have run either one, so neither leftover is assumed and neither is left out.
            leftover = ("It names no runtime, so check both: that no container was left behind (`docker ps -a "
                        "--filter name=llmbench-`) and that no llama-server it started is still running (`pgrep -fl "
                        "llama-server`), then delete the file")
        return f"{target} is held by {who}{since}, which is no longer running: the lock is stale. {leftover}"
    state = "is still running" if alive else "could not be checked"
    server = f"; its llama-server is pid {server_pid}" if server_pid is not None else ""
    return (f"{target} is held by {who}{since}, which {state} (a reused pid can belong to an unrelated process); "
            f"one GPU workload runs at a time - wait for it, or stop that run first{server}")
