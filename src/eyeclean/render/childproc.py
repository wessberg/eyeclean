"""Lifetime control for the headless child processes (Blender) this CLI starts.

Ctrl+C raises ``KeyboardInterrupt`` in the main thread only, and the render workers run in a thread
pool, so ``subprocess.run``'s own kill-on-interrupt never fires for them: without this the children
keep rendering unattended long after the CLI stopped reporting progress.  Every child is registered
here so a single call terminates all of them, and on Windows each is also assigned to a
kill-on-close job object, so they die with this process even when it is killed outright or its
console window is closed.

Children are killed, not asked to stop: a half-written shot leaves no ``meta.json``, which is
exactly what makes the next ``eyeclean generate`` re-render it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading

_LOCK = threading.RLock()  # reentrant: an interrupt can land in a thread that already holds it
_LIVE: dict[int, subprocess.Popen] = {}
_CANCELLED = threading.Event()


class Cancelled(RuntimeError):
    """Raised instead of starting a child once the run has been cancelled."""


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    # private DLL instance, so these argtypes do not leak into other ctypes users of kernel32
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KILL_ON_JOB_CLOSE, _EXTENDED_LIMIT_INFORMATION = 0x2000, 9

    class _BasicLimits(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):  # IO_COUNTERS
        _fields_ = [(n, ctypes.c_uint64) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimits(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _kill_on_close_job():
        """A job object whose processes Windows terminates when the last handle to it closes.

        The handle is deliberately never closed while we run: it has to outlive every child, and
        letting the OS drop it as this process dies is the whole point.  Nested jobs are fine on
        Windows 8+, so being inside someone else's job (a CI runner, a launcher) is not a problem.
        """
        h = _k32.CreateJobObjectW(None, None)
        if not h:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(h, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
            _k32.CloseHandle(h)
            return None
        return h

    _JOB = _kill_on_close_job()

    def _adopt(proc: subprocess.Popen) -> None:
        # Popen._handle is the handle CPython already holds, so there is no pid-reuse window here.
        # Best effort: an unadoptable child is still killed by cancel_all(), just not by our death.
        if _JOB:
            _k32.AssignProcessToJobObject(_JOB, int(proc._handle))
else:
    def _adopt(proc: subprocess.Popen) -> None:
        return None


def cancelled() -> bool:
    return _CANCELLED.is_set()


def reset() -> None:
    """Allow children again: a command starting fresh must not inherit an earlier run's cancel."""
    _CANCELLED.clear()


def cancel_all() -> int:
    """Terminate every live child and refuse new ones; returns how many were still running.

    Safe to call from any thread: it only touches handles this module owns.
    """
    _CANCELLED.set()
    with _LOCK:
        procs = list(_LIVE.values())
    n = 0
    for p in procs:
        try:
            p.kill()
            n += 1
        except OSError:  # already exited between the snapshot and here
            pass
    return n


@contextlib.contextmanager
def spawn(cmd: list[str], **kw):
    """Start ``cmd`` as a tracked child; kills it on the way out however the block ends."""
    with _LOCK:  # the same lock cancel_all() takes, so a cancel cannot slip past a starting child
        if _CANCELLED.is_set():
            raise Cancelled(f"run cancelled; not starting {os.path.basename(cmd[0])}")
        proc = subprocess.Popen(cmd, **kw)
        _adopt(proc)
        _LIVE[proc.pid] = proc
    try:
        yield proc
    finally:
        with _LOCK:
            _LIVE.pop(proc.pid, None)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
