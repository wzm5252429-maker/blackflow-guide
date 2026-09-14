"""Physical-pixel Windows input, bound to the captured game window."""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
import os

from .capture import physical_pixel_context


class WindowsController:
    def __init__(self, capture):
        if os.name != 'nt':
            raise RuntimeError('游戏接管需要 Windows 10/11')
        self.capture = capture
        self.user32 = ct.WinDLL('user32', use_last_error=True)
        for name, args, result in [
            ('SetForegroundWindow', [wt.HWND], wt.BOOL),
            ('GetForegroundWindow', [], wt.HWND),
            ('GetAncestor', [wt.HWND, wt.UINT], wt.HWND),
            ('ShowWindow', [wt.HWND, ct.c_int], wt.BOOL),
            ('IsIconic', [wt.HWND], wt.BOOL),
            ('GetCursorPos', [ct.POINTER(wt.POINT)], wt.BOOL),
            ('GetAsyncKeyState', [ct.c_int], ct.c_short),
            ('GetSystemMetrics', [ct.c_int], ct.c_int),
            ('WindowFromPoint', [wt.POINT], wt.HWND),
        ]:
            fn = getattr(self.user32, name)
            fn.argtypes, fn.restype = args, result

    def focus(self):
        with physical_pixel_context():
            self._focus_physical()

    def _focus_physical(self):
        hwnd = self.capture.current_geometry().hwnd
        if self.user32.IsIconic(hwnd):
            self.user32.ShowWindow(hwnd, 9)
        self.user32.SetForegroundWindow(hwnd)
        if self.user32.GetAncestor(self.user32.GetForegroundWindow(), 2) != hwnd:
            raise RuntimeError('未能聚焦游戏窗口。请切换到游戏，或以与游戏相同的权限启动接管器')

    def emergency_stop(self):
        with physical_pixel_context():
            return self._emergency_stop_physical()

    def _emergency_stop_physical(self):
        if self.user32.GetAsyncKeyState(0x1B) & 0x8000:
            return True
        point = wt.POINT()
        if self.user32.GetCursorPos(ct.byref(point)):
            left, top = self.user32.GetSystemMetrics(76), self.user32.GetSystemMetrics(77)
            return point.x <= left + 2 and point.y <= top + 2
        return False

    def click(self, action, frame):
        # Win32 virtualizes several of these APIs independently of how capture
        # was obtained. Keep hit-testing, desktop metrics and SendInput in the
        # same physical-pixel context as the captured geometry, including on
        # mixed-DPI monitors and when the Python host is already DPI-unaware.
        with physical_pixel_context():
            self._click_physical(action, frame)

    def _click_physical(self, action, frame):
        self.capture.assert_geometry_current(frame.geometry)
        hwnd = frame.geometry.hwnd
        if self.user32.GetAncestor(self.user32.GetForegroundWindow(), 2) != hwnd:
            raise RuntimeError('游戏已失去前台焦点，接管停止；请返回游戏后重新启动')
        x, y, width, height = action.bbox
        # Reject boxes that straddle normalization padding as well as their center.
        frame.transform.recognition_to_client(x, y)
        frame.transform.recognition_to_client(x + width - .01, y + height - .01)
        sx, sy = frame.transform.recognition_to_screen(x + width / 2, y + height / 2)
        sx, sy = round(sx), round(sy)
        if not frame.geometry.client_rect.contains(sx, sy):
            raise RuntimeError('点击位置超出游戏客户区，正在等待新的截图')
        point = wt.POINT(sx, sy)
        owner = self.user32.GetAncestor(self.user32.WindowFromPoint(point), 2)
        if owner != hwnd:
            raise RuntimeError('点击位置被其他窗口遮挡，接管停止')
        left, top = self.user32.GetSystemMetrics(76), self.user32.GetSystemMetrics(77)
        sw, sh = self.user32.GetSystemMetrics(78), self.user32.GetSystemMetrics(79)
        if sw < 2 or sh < 2:
            raise RuntimeError('无法获取有效的物理桌面尺寸')
        if not left <= sx < left + sw or not top <= sy < top + sh:
            raise RuntimeError('点击位置不在当前桌面范围内')

        class MouseInput(ct.Structure):
            _fields_ = [('dx', wt.LONG), ('dy', wt.LONG), ('mouseData', wt.DWORD),
                        ('dwFlags', wt.DWORD), ('time', wt.DWORD), ('dwExtraInfo', ct.c_size_t)]
        class KeyboardInput(ct.Structure):
            _fields_ = [('wVk', wt.WORD), ('wScan', wt.WORD), ('dwFlags', wt.DWORD),
                        ('time', wt.DWORD), ('dwExtraInfo', ct.c_size_t)]
        class HardwareInput(ct.Structure):
            _fields_ = [('uMsg', wt.DWORD), ('wParamL', wt.WORD), ('wParamH', wt.WORD)]
        class InputUnion(ct.Union):
            _fields_ = [('mi', MouseInput), ('ki', KeyboardInput), ('hi', HardwareInput)]
        class Input(ct.Structure):
            _anonymous_ = ('value',)
            _fields_ = [('type', wt.DWORD), ('value', InputUnion)]
        move = MouseInput(round((sx-left)*65535/(sw-1)), round((sy-top)*65535/(sh-1)),
                          0, 0x8000 | 0x4000 | 0x0001, 0, 0)
        inputs = (Input * 3)(Input(0, InputUnion(mi=move)),
                              Input(0, InputUnion(mi=MouseInput(0, 0, 0, 2, 0, 0))),
                              Input(0, InputUnion(mi=MouseInput(0, 0, 0, 4, 0, 0))))
        send = self.user32.SendInput
        send.argtypes, send.restype = [wt.UINT, ct.POINTER(Input), ct.c_int], wt.UINT
        # Recheck after constructing the batch too: hit-testing and structure
        # setup must not leave a move/resize between validation and submission.
        self.capture.assert_geometry_current(frame.geometry)
        if self.user32.GetAncestor(self.user32.GetForegroundWindow(), 2) != hwnd:
            raise RuntimeError('提交点击前游戏失去前台焦点，接管停止')
        if send(3, inputs, ct.sizeof(Input)) != 3:
            # Always attempt a release if Windows rejected a partially submitted batch.
            release = Input(0, InputUnion(mi=MouseInput(0, 0, 0, 4, 0, 0)))
            send(1, ct.byref(release), ct.sizeof(Input))
            raise RuntimeError('Windows 拒绝输入，请使用与游戏相同的运行权限')
