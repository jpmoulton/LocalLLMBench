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


def describe_lock(path: str | Path) -> str:
    """One sentence for an error message: the holder, whether it is still running, and what to do about it."""
    target = Path(path)
    try:
        holder = json.loads(target.read_bytes()[:4096].decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        holder = None
    if not isinstance(holder, dict) or type(holder.get("pid")) is not int:
        return (f"{target} exists but does not name its holder; confirm no llmbench run is active before "
                "deleting it")
    pid, alive = holder["pid"], process_alive(holder["pid"])
    who = f"pid {pid}" + (f" ({holder['owner']})" if isinstance(holder.get("owner"), str) else "")
    since = f", created {holder['created']}" if isinstance(holder.get("created"), str) else ""
    if alive is False:
        return (f"{target} is held by {who}{since}, which is no longer running: the lock is stale. Check that no "
                f"container was left behind (`docker ps -a --filter name=llmbench-`), then delete the file")
    state = "is still running" if alive else "could not be checked"
    return (f"{target} is held by {who}{since}, which {state} (a reused pid can belong to an unrelated process); "
            "one GPU workload runs at a time - wait for it, or stop that run first")
