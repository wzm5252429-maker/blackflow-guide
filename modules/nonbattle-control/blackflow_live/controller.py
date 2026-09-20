"""Physical-pixel Windows input, bound to the captured game window."""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
import os
import time

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
        self.cancel_pointer_cleanup()
        window = self.capture.verified_window()
        hwnd = window.hwnd
        if not window.visible:
            raise RuntimeError('明日方舟窗口已隐藏，请先从游戏启动器打开游戏窗口')
        if window.minimized:
            self.user32.ShowWindow(hwnd, 9)
        # Restoration changes client coordinates. Verify the selected process
        # again and read the restored geometry before attempting activation.
        self.capture.current_geometry()
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
        self.cancel_pointer_cleanup()
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
        # Do not append a move to this click batch. Unity can sample the mouse
        # position on its next frame instead of consuming every event position.
        def move_only(px, py, desktop):
            dl,dt,dw,dh = desktop
            motion=MouseInput(round((px-dl)*65535/(dw-1)),round((py-dt)*65535/(dh-1)),
                              0,0x8000 | 0x4000 | 0x0001,0,0)
            batch=(Input * 1)(Input(0,InputUnion(mi=motion)))
            return send(1,batch,ct.sizeof(Input)) == 1
        self._pointer_cleanup = (frame.geometry,(sx,sy),time.monotonic(),time.time(),move_only)

    def cancel_pointer_cleanup(self):
        """Forget a deferred move without interacting with Windows."""
        self._pointer_cleanup = None

    def clear_pointer(self, frame, *, still_active=None):
        """Best-effort MOVE-only cleanup after a new nonbattle observation.

        The engine owns the active-session/scene checks. Preview never calls
        this method. A rejected cleanup does not undo a successful click.
        """
        with physical_pixel_context():
            pending=getattr(self,'_pointer_cleanup',None)
            if pending is None:
                return False
            geometry,clicked,submitted,wall_time,move_only=pending
            elapsed=time.monotonic()-submitted
            if 0 <= elapsed < .15:
                return False
            self.cancel_pointer_cleanup()
            if still_active is not None and not still_active():
                return False
            if elapsed < 0 or elapsed > 15 or frame.captured_at <= wall_time:
                return False
            if frame.geometry.identity != geometry.identity:
                return False
            try:
                def unchanged_pointer():
                    if self._emergency_stop_physical():
                        return False
                    # MOVE with a user-held button would become a drag.
                    if any(self.user32.GetAsyncKeyState(key)&0x8000 for key in (1,2,4,5,6)):
                        return False
                    point=wt.POINT()
                    return bool(self.user32.GetCursorPos(ct.byref(point))
                                and abs(point.x-clicked[0]) <= 3 and abs(point.y-clicked[1]) <= 3)
                if not unchanged_pointer():
                    return False
                self.capture.assert_geometry_current(geometry)
                hwnd=geometry.hwnd
                if self.user32.GetAncestor(self.user32.GetForegroundWindow(),2) != hwnd:
                    return False
                left,top=self.user32.GetSystemMetrics(76),self.user32.GetSystemMetrics(77)
                width,height=self.user32.GetSystemMetrics(78),self.user32.GetSystemMetrics(79)
                if width < 2 or height < 2:
                    return False
                rect=geometry.client_rect
                px,py=int(min(rect.right,left+width))-1,int(min(rect.bottom,top+height))-1
                if not rect.contains(px,py) or not (left<=px<left+width and top<=py<top+height):
                    return False
                if self.user32.GetAncestor(self.user32.WindowFromPoint(wt.POINT(px,py)),2) != hwnd:
                    return False
                self.capture.assert_geometry_current(geometry)
                if (self.user32.GetAncestor(self.user32.GetForegroundWindow(),2) != hwnd
                    or not unchanged_pointer()):
                    return False
                if self.user32.GetAncestor(self.user32.WindowFromPoint(wt.POINT(px,py)),2) != hwnd:
                    return False
                if still_active is not None and not still_active():
                    return False
                if tuple(self.user32.GetSystemMetrics(index) for index in (76,77,78,79)) != (left,top,width,height):
                    return False
                return move_only(px,py,(left,top,width,height))
            except (OSError,RuntimeError):
                return False
