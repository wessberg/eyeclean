"""Pause/resume hotkey for long console runs: press p to pause, p again to resume.

Pausing suspends every process this one started (the headless Blender workers), so their GPU and
CPU work stops at once; their GPU memory stays allocated.  `wait()` holds this process's own work
(texture synthesis, cache conversion, launching the next worker) at its next checkpoint.  Ctrl+C
or closing the console resumes the workers first and then runs ``on_interrupt`` (the render
stage passes ``childproc.cancel_all``), so no worker is left running behind the CLI.
Only active on an interactive Windows console; otherwise nothing is suspended and `wait()` never blocks.
"""

from __future__ import annotations

import os
import sys
import threading
import time

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    # private DLL instances, so these argtypes do not leak into other ctypes users of kernel32
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    _SUSPEND_RESUME, _QUERY_LIMITED, _SNAPPROCESS = 0x0800, 0x1000, 0x2

    class _ProcessEntry(ctypes.Structure):  # PROCESSENTRY32W
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    _CtrlHandler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.SetConsoleCtrlHandler.argtypes = [_CtrlHandler, wintypes.BOOL]
    _ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    def _created(h) -> int | None:
        t = [wintypes.FILETIME() for _ in range(4)]
        if not _k32.GetProcessTimes(h, *map(ctypes.byref, t)):
            return None
        return t[0].dwHighDateTime << 32 | t[0].dwLowDateTime

    def _children_by_parent() -> dict[int, list[int]]:
        snap = _k32.CreateToolhelp32Snapshot(_SNAPPROCESS, 0)
        if snap in (None, wintypes.HANDLE(-1).value):
            return {}
        out: dict[int, list[int]] = {}
        try:
            e = _ProcessEntry(dwSize=ctypes.sizeof(_ProcessEntry))
            ok = _k32.Process32FirstW(snap, ctypes.byref(e))
            while ok:
                out.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
                ok = _k32.Process32NextW(snap, ctypes.byref(e))
        finally:
            _k32.CloseHandle(snap)
        return out


class KeyboardPause:
    """`with KeyboardPause(tag="generate") as pause: work(pause=pause)`; the work calls `pause.wait()`."""

    def __init__(self, log=print, key: str = "p", tag: str = "pause", on_interrupt=None):
        self.log, self.key, self.tag, self.on_interrupt = log, key, tag, on_interrupt
        self.enabled = os.name == "nt" and sys.stdin is not None and sys.stdin.isatty()
        self._running = threading.Event()
        self._running.set()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._held: dict[int, tuple[int, int]] = {}  # pid -> (open handle, creation time) of suspended processes
        self._since = 0.0
        self._paused_total = 0.0
        self._thread: threading.Thread | None = None
        self._ctrl = None

    def __enter__(self) -> KeyboardPause:
        if self.enabled:
            self._ctrl = _CtrlHandler(self._on_ctrl)  # keep a reference: the OS calls it from its own thread
            _k32.SetConsoleCtrlHandler(self._ctrl, True)
            self._thread = threading.Thread(target=self._listen, name="pause-hotkey", daemon=True)
            self._thread.start()
            self.log(f"[{self.tag}] press {self.key} to pause/resume")
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            if self.paused:
                self._resume()
        if self._ctrl is not None:
            _k32.SetConsoleCtrlHandler(self._ctrl, False)

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def wait(self) -> None:
        """Checkpoint for the caller's own work: blocks while paused."""
        while not self._running.wait(0.5):  # timed, so Ctrl+C still reaches the main thread
            pass

    def paused_seconds(self) -> float:
        """Total time spent paused, including a pause in progress."""
        with self._lock:
            return self._paused_total + (time.monotonic() - self._since if self.paused else 0.0)

    def toggle(self) -> None:
        with self._lock:
            if self.paused:
                self._resume()
            else:
                self._running.clear()
                self._since = time.monotonic()
                n = self._suspend_workers()
                self.log(f"[{self.tag}] PAUSED - {n} worker process(es) suspended, their GPU memory stays allocated; "
                         f"press {self.key} to resume")

    def _resume(self) -> None:
        for h, _ in self._held.values():
            _ntdll.NtResumeProcess(h)
            _k32.CloseHandle(h)
        self._held.clear()
        dt = time.monotonic() - self._since
        self._paused_total += dt
        self._running.set()
        self.log(f"[{self.tag}] resumed after {dt:.0f}s")

    def _suspend_workers(self) -> int:
        """Suspend every descendant process not suspended yet; returns how many are suspended in total."""
        if os.name != "nt":
            return 0
        kids = _children_by_parent()
        todo, seen = [(os.getpid(), _created(_k32.GetCurrentProcess()))], set()
        while todo:
            pid, born = todo.pop()
            for c in kids.get(pid, ()):
                if c in seen:
                    continue
                seen.add(c)
                if c in self._held:
                    todo.append((c, self._held[c][1]))
                    continue
                h = _k32.OpenProcess(_SUSPEND_RESUME | _QUERY_LIMITED, False, c)
                if not h:
                    continue
                t = _created(h)
                # Windows keeps a dead parent's pid in its orphans, so a process older than this "parent"
                # belongs to whoever had the pid before: never ours
                if t is None or born is None or t < born or _ntdll.NtSuspendProcess(h) != 0:
                    _k32.CloseHandle(h)
                    continue
                self._held[c] = (h, t)
                todo.append((c, t))
        return len(self._held)

    def _listen(self) -> None:
        rescan = 0.0
        while not self._stop.wait(0.1):
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # function and arrow keys arrive as two codes
                    msvcrt.getwch()
                elif ch.lower() == self.key:
                    self.toggle()
            if self.paused and time.monotonic() > rescan:  # a worker that was starting just as we paused
                with self._lock:
                    if self.paused:
                        self._suspend_workers()
                rescan = time.monotonic() + 1.0

    def _on_ctrl(self, event: int) -> bool:
        """Ctrl+C / Ctrl+Break / console closed: resume the workers, then stop them for good.

        Windows calls this on its own thread the moment the key is pressed, so it is the only
        place that can act immediately: the main thread may be parked in a lock it will not leave
        until a worker finishes, and a worker thread never receives the interrupt at all.
        """
        with self._lock:
            if self.paused:
                self._resume()
        if self.on_interrupt is not None:
            try:
                self.on_interrupt()
            except Exception as e:  # never let cleanup swallow the interrupt itself
                self.log(f"[{self.tag}] interrupt cleanup failed: {e}")
        return False  # continue with the default handling (KeyboardInterrupt)
