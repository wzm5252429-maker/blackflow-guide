"""Read-only MAA FramePool capture with per-frame physical Win32 geometry.

This bypasses MaaCore's 16:9 screenshot proxy and calls the raw capture unit.
The cv::Mat / ControlUnit ABI is specific to Windows x64 MAA release binaries;
the bundled BFMapRecognizer v1.0.2 uses v6.17.0-beta.6. Do not substitute an
unrelated OpenCV DLL or a different ControlUnit build. No mouse or keyboard
method is enabled on this controller, and capture never focuses/restores a game.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time

import numpy as np

from .geometry import CapturedFrame, Rect, WindowGeometry


DEFAULT_MAA_RUNTIME = Path("D:/ArknightsAuto/BFMapRecognizer_v1.0.2_Windows/BFMapRecognizer")


def _user32():
    if os.name != "nt":
        raise RuntimeError("Game window capture requires Windows")
    user = ct.WinDLL("user32", use_last_error=True)
    user.IsWindow.argtypes = [wt.HWND]
    user.IsWindowVisible.argtypes = [wt.HWND]
    user.IsIconic.argtypes = [wt.HWND]
    user.GetClientRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
    user.ClientToScreen.argtypes = [wt.HWND, ct.POINTER(wt.POINT)]
    user.GetWindowThreadProcessId.argtypes = [wt.HWND, ct.POINTER(wt.DWORD)]
    user.GetWindowTextLengthW.argtypes = [wt.HWND]
    user.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ct.c_int]
    user.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ct.c_int]
    user.GetDpiForWindow.argtypes = [wt.HWND]
    user.GetDpiForWindow.restype = wt.UINT
    user.SetThreadDpiAwarenessContext.argtypes = [ct.c_void_p]
    user.SetThreadDpiAwarenessContext.restype = ct.c_void_p
    return user


@contextmanager
def physical_pixel_context():
    """Use per-monitor physical coordinates without changing the host process."""
    user = _user32()
    previous = user.SetThreadDpiAwarenessContext(ct.c_void_p(-4))
    if not previous:
        raise OSError(ct.get_last_error(), "Cannot enable per-monitor DPI coordinates")
    try:
        yield user
    finally:
        user.SetThreadDpiAwarenessContext(previous)


def window_geometry(hwnd: int) -> WindowGeometry:
    with physical_pixel_context() as user:
        if not hwnd or not user.IsWindow(hwnd):
            raise RuntimeError("所选游戏窗口已关闭，请重新连接游戏窗口")
        if not user.IsWindowVisible(hwnd):
            raise RuntimeError("游戏窗口处于隐藏状态，请先显示游戏窗口后重试")
        if user.IsIconic(hwnd):
            raise RuntimeError("游戏窗口已最小化，请先恢复游戏窗口后重试识别预览")
        rect, origin, pid = wt.RECT(), wt.POINT(0, 0), wt.DWORD()
        if not user.GetClientRect(hwnd, ct.byref(rect)) or not user.ClientToScreen(hwnd, ct.byref(origin)):
            raise OSError(ct.get_last_error(), "Cannot read game client geometry")
        if not user.GetWindowThreadProcessId(hwnd, ct.byref(pid)):
            raise OSError(ct.get_last_error(), "Cannot identify the game process")
        length = user.GetWindowTextLengthW(hwnd)
        title = ct.create_unicode_buffer(length + 1)
        user.GetWindowTextW(hwnd, title, length + 1)
        dpi = int(user.GetDpiForWindow(hwnd))
        return WindowGeometry(
            int(hwnd), int(pid.value), title.value,
            Rect(origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top), dpi or 96,
        )


def _process_snapshot() -> dict[int, str]:
    """Read executable basenames without opening protected game processes."""
    class ProcessEntry(ct.Structure):
        _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD),
                    ("th32ProcessID", wt.DWORD), ("th32DefaultHeapID", ct.c_size_t),
                    ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                    ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", wt.LONG),
                    ("dwFlags", wt.DWORD), ("szExeFile", wt.WCHAR * 260)]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    kernel.Process32FirstW.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
    kernel.Process32FirstW.restype = wt.BOOL
    kernel.Process32NextW.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
    kernel.Process32NextW.restype = wt.BOOL
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    handle = kernel.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if not handle or handle == ct.c_void_p(-1).value:
        return {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ct.sizeof(entry)
        result = {}
        present = kernel.Process32FirstW(handle, ct.byref(entry))
        while present:
            result[int(entry.th32ProcessID)] = entry.szExeFile.casefold()
            present = kernel.Process32NextW(handle, ct.byref(entry))
        return result
    finally:
        kernel.CloseHandle(handle)


def _process_basename(pid: int) -> str | None:
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wt.BOOL
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        try:
            size = wt.DWORD(32768)
            name = ct.create_unicode_buffer(size.value)
            if kernel.QueryFullProcessImageNameW(handle, 0, name, ct.byref(size)):
                return name.value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        finally:
            kernel.CloseHandle(handle)
    # A protected/elevated game may deny an image-path query while Windows'
    # read-only process snapshot still identifies its executable by PID.
    return _process_snapshot().get(int(pid))


@dataclass(frozen=True)
class GameWindowCandidate:
    """Verified identity and visibility, without requiring capturable geometry."""
    hwnd: int
    pid: int
    title: str
    class_name: str
    visible: bool
    minimized: bool
    executable: str = "arknights.exe"

    @property
    def state(self):
        if not self.visible:
            return "hidden"
        return "minimized" if self.minimized else "visible"

    def to_dict(self):
        return {**asdict(self), "state": self.state}


def find_game_candidates() -> list[GameWindowCandidate]:
    """Find verified Unity game clients, including minimized/hidden clients.

    Titles are descriptive only: browser titles and the game's hidden Qt
    helpers are never accepted as render clients. This function reads metadata
    only; it never restores, focuses, captures, or sends input to a window.
    """
    user = _user32()
    matches = []
    processes = {}
    callback_type = getattr(ct, "WINFUNCTYPE", ct.CFUNCTYPE)(wt.BOOL, wt.HWND, wt.LPARAM)

    @callback_type
    def visit(hwnd, _):
        try:
            if not user.IsWindow(hwnd):
                return True
            class_name = ct.create_unicode_buffer(256)
            if not user.GetClassNameW(hwnd, class_name, len(class_name)) or class_name.value != "UnityWndClass":
                return True
            pid = wt.DWORD()
            if not user.GetWindowThreadProcessId(hwnd, ct.byref(pid)) or not pid.value:
                return True
            process_id = int(pid.value)
            if process_id not in processes:
                processes[process_id] = _process_basename(process_id)
            executable = processes[process_id]
            if executable != "arknights.exe":
                return True
            title = ct.create_unicode_buffer(user.GetWindowTextLengthW(hwnd) + 1)
            user.GetWindowTextW(hwnd, title, len(title))
            # The HWND may have disappeared/reused while reading its process.
            if not user.GetWindowThreadProcessId(hwnd, ct.byref(pid)) or pid.value != process_id:
                return True
            matches.append(GameWindowCandidate(
                int(hwnd), process_id, title.value, class_name.value,
                bool(user.IsWindowVisible(hwnd)), bool(user.IsIconic(hwnd)), executable,
            ))
        except (OSError, RuntimeError, ValueError):
            pass
        return True

    user.EnumWindows.argtypes = [callback_type, wt.LPARAM]
    user.EnumWindows.restype = wt.BOOL
    if not user.EnumWindows(visit, 0):
        raise OSError(ct.get_last_error(), "Cannot enumerate game windows")
    return sorted(matches, key=lambda item: (item.pid, item.hwnd))


def find_game_windows() -> list[WindowGeometry]:
    """Compatibility view containing only verified, currently usable clients."""
    matches = []
    for item in find_game_candidates():
        if not item.visible or item.minimized:
            continue
        try:
            geometry = window_geometry(item.hwnd)
            if geometry.pid == item.pid:
                matches.append(geometry)
        except (OSError, RuntimeError, ValueError):
            pass
    return matches


def assert_geometry_current(snapshot: WindowGeometry) -> WindowGeometry:
    current = window_geometry(snapshot.hwnd)
    if current.identity != snapshot.identity:
        raise RuntimeError("Game window moved, resized, changed DPI, or changed process; recapture before input")
    return current


def resolve_runtime(runtime=None) -> Path:
    path = Path(runtime or os.environ.get("BLACKFLOW_MAA_RUNTIME") or DEFAULT_MAA_RUNTIME).resolve()
    missing = [name for name in ("MaaWin32ControlUnit.dll", "opencv_world4_maa.dll") if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"MAA capture runtime is missing {', '.join(missing)} in {path}")
    return path


class MaaWindowCapture:
    """Minimal raw ControlUnit ABI adapter; no input is configured or exposed."""
    def __init__(self, runtime: Path, hwnd: int):
        if os.name != "nt" or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("MAA raw capture requires Windows x64")
        self.runtime = resolve_runtime(runtime)
        self._unit = None
        self._dll_dir = os.add_dll_directory(str(self.runtime))
        try:
            self._control = ct.WinDLL(str(self.runtime / "MaaWin32ControlUnit.dll"))
            self._opencv = ct.WinDLL(str(self.runtime / "opencv_world4_maa.dll"))
            self._create = self._control.MaaWin32ControlUnitCreate
            self._create.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_uint64, ct.c_uint64]
            self._create.restype = ct.c_void_p
            self._destroy = self._control.MaaWin32ControlUnitDestroy
            self._destroy.argtypes = [ct.c_void_p]
            self._destroy.restype = None
            self._ctor = getattr(self._opencv, "??0Mat@cv@@QEAA@XZ")
            self._ctor.argtypes = [ct.c_void_p]
            self._ctor.restype = ct.c_void_p
            self._dtor = getattr(self._opencv, "??1Mat@cv@@QEAA@XZ")
            self._dtor.argtypes = [ct.c_void_p]
            self._dtor.restype = None
            self._unit = self._create(hwnd, 2, 0, 0)  # FramePool, no mouse, no keyboard.
            if not self._unit:
                raise RuntimeError("MAA could not create a FramePool capture unit")
            table = ct.cast(self._unit, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
            connect = ct.WINFUNCTYPE(ct.c_bool, ct.c_void_p)(table[1])
            self._capture = ct.WINFUNCTYPE(ct.c_bool, ct.c_void_p, ct.c_void_p)(table[7])
            if not connect(self._unit):
                raise RuntimeError("MAA FramePool could not connect to the selected game window")
        except BaseException:
            self.close()
            raise

    def capture(self) -> np.ndarray:
        if not self._unit:
            raise RuntimeError("Capture controller is closed")
        storage = ct.create_string_buffer(128)
        address = ct.addressof(storage)
        self._ctor(address)
        try:
            if not self._capture(self._unit, address):
                raise RuntimeError("MAA FramePool screenshot failed")
            return mat_to_bgr(address)
        finally:
            self._dtor(address)

    def close(self):
        if self._unit:
            self._destroy(self._unit)
            self._unit = None
        if self._dll_dir:
            self._dll_dir.close()
            self._dll_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def mat_to_bgr(address: int) -> np.ndarray:
    """Copy a checked release-ABI cv::Mat, retaining no native-owned pointers."""
    flags, dims, rows, cols = (ct.c_int.from_address(address + n).value for n in (0, 4, 8, 12))
    data = ct.c_void_p.from_address(address + 16).value
    step_ptr = ct.c_void_p.from_address(address + 72).value
    if dims != 2 or flags & 0xFFF != 16 or not data or not step_ptr:
        raise RuntimeError("Unexpected native screenshot ABI or pixel format (expected BGR8)")
    if not (1 <= rows <= 16384 and 1 <= cols <= 16384):
        raise RuntimeError("Invalid native screenshot dimensions")
    stride = ct.c_size_t.from_address(step_ptr).value
    if not cols * 3 <= stride <= cols * 3 + 65536:
        raise RuntimeError("Invalid native screenshot stride")
    raw = ct.string_at(data, stride * rows)
    return np.frombuffer(raw, dtype=np.uint8).reshape(rows, stride)[:, :cols * 3].reshape(rows, cols, 3).copy()


class WindowsGameCapture:
    def __init__(self, runtime=None, hwnd: int | None = None):
        self.runtime = resolve_runtime(runtime)
        self.hwnd = int(hwnd) if hwnd else None
        self._selected_pid = None
        self._native = None

    @staticmethod
    def discover_windows():
        result = []
        for candidate in find_game_candidates():
            item = candidate.to_dict()
            if candidate.visible and not candidate.minimized:
                try:
                    geometry = window_geometry(candidate.hwnd)
                    if geometry.pid != candidate.pid:
                        continue
                    item.update(geometry.to_dict())
                except (OSError, RuntimeError, ValueError):
                    # Keep the verified identity for a useful state/error in the
                    # next attempt; do not invent geometry for an unavailable frame.
                    pass
            result.append(item)
        return result

    def select_window(self, hwnd: int):
        candidates = {item.hwnd: item for item in find_game_candidates()}
        if int(hwnd) not in candidates:
            raise RuntimeError("所选窗口不是已验证的明日方舟游戏画面窗口，请重新选择")
        self.close()
        self.hwnd = int(hwnd)
        self._selected_pid = candidates[self.hwnd].pid
        return candidates[self.hwnd]

    def verified_window(self) -> GameWindowCandidate:
        """Select/revalidate identity without requiring or changing visibility."""
        candidates = find_game_candidates()
        if self.hwnd is None:
            if not candidates:
                if "arknights.exe" in _process_snapshot().values():
                    raise RuntimeError("检测到明日方舟进程，但未找到游戏画面窗口；请等待客户端加载完成并显示游戏窗口")
                raise RuntimeError("未找到明日方舟游戏窗口，请先启动游戏客户端并等待游戏画面出现")
            if len(candidates) > 1:
                raise RuntimeError("检测到多个明日方舟游戏窗口，请先在网页中选择一个窗口")
            self.hwnd = candidates[0].hwnd
        candidate = next((item for item in candidates if item.hwnd == self.hwnd), None)
        if candidate is None:
            self.close()
            raise RuntimeError("所选游戏窗口已关闭或身份已改变，请重新连接游戏窗口")
        if self._selected_pid is not None and candidate.pid != self._selected_pid:
            self.close()
            raise RuntimeError("所选游戏窗口的进程已改变，请重新连接游戏窗口")
        self._selected_pid = candidate.pid
        return candidate

    def current_geometry(self):
        candidate = self.verified_window()
        if not candidate.visible:
            raise RuntimeError("游戏窗口处于隐藏状态，请先显示游戏窗口后重试")
        if candidate.minimized:
            raise RuntimeError("游戏窗口已最小化，请先恢复游戏窗口后重试识别预览")
        geometry = window_geometry(candidate.hwnd)
        if geometry.pid != candidate.pid:
            self.close()
            raise RuntimeError("读取期间游戏窗口的进程已改变，请重新连接游戏窗口")
        return geometry

    def capture(self) -> CapturedFrame:
        # Re-read before/after every frame. Moving the game while the capture is
        # in flight must never produce a click using an old desktop origin.
        for _ in range(3):
            before = self.current_geometry()
            if self._native is None:
                # current_geometry verifies both explicit HWNDs and the pinned
                # process before any native controller can be created.
                with physical_pixel_context():
                    self._native = MaaWindowCapture(self.runtime, self.hwnd)
            captured_at = time.time()
            with physical_pixel_context():
                pixels = self._native.capture()
            after = self.current_geometry()
            if before.identity != after.identity or pixels.shape[:2] != (after.height, after.width):
                # Recreate FramePool on resize; its first frame may still have
                # the old dimensions. Never infer/stretch a mismatched capture.
                self.close()
                continue
            return CapturedFrame(pixels, after, captured_at=captured_at)
        raise RuntimeError("Game geometry did not stabilize or raw capture did not match its physical client")

    def capture_normalized(self, target_size=(1280, 720), **kwargs):
        return self.capture().normalize(target_size, **kwargs)

    def assert_geometry_current(self, snapshot):
        candidate = self.verified_window()
        if candidate.hwnd != snapshot.hwnd or candidate.pid != snapshot.pid:
            raise RuntimeError("游戏窗口身份已改变，必须重新截图后才能点击")
        return assert_geometry_current(snapshot)

    def close(self):
        if self._native is not None:
            self._native.close()
            self._native = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


WindowsCapture = WindowsGameCapture
