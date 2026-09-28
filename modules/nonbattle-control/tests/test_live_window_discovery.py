"""Window-discovery regressions using metadata mocks, never real game access."""
import ctypes as ct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from blackflow_live.capture import (
    GameWindowCandidate, WindowsGameCapture, _process_basename,
    _process_snapshot, find_game_candidates, find_game_windows,
)
from blackflow_live.geometry import Rect, WindowGeometry


def candidate(hwnd=123, pid=42, *, visible=True, minimized=False, title="明日方舟"):
    return GameWindowCandidate(hwnd, pid, title, "UnityWndClass", visible, minimized)


def geometry(hwnd=123, pid=42):
    return WindowGeometry(hwnd, pid, "明日方舟", Rect(0, 0, 2878, 1659), 192)


def fake_user(records):
    """Expose only metadata APIs; any attempt to capture/focus/input fails."""
    def text(hwnd, buffer, size):
        buffer.value = records[int(hwnd)].get("title", "明日方舟")
        return len(buffer.value)

    def class_name(hwnd, buffer, size):
        buffer.value = records[int(hwnd)].get("class_name", "UnityWndClass")
        return len(buffer.value)

    def process(hwnd, pointer):
        pointer._obj.value = records[int(hwnd)].get("pid", 42)
        return 1

    def enumerate_windows(callback, data):
        for hwnd in records:
            if not callback(hwnd, data):
                return False
        return True

    return SimpleNamespace(
        IsWindow=Mock(side_effect=lambda hwnd: int(hwnd) in records),
        IsWindowVisible=Mock(side_effect=lambda hwnd: records[int(hwnd)].get("visible", True)),
        IsIconic=Mock(side_effect=lambda hwnd: records[int(hwnd)].get("minimized", False)),
        GetWindowTextLengthW=Mock(side_effect=lambda hwnd: len(records[int(hwnd)].get("title", "明日方舟"))),
        GetWindowTextW=Mock(side_effect=text),
        GetClassNameW=Mock(side_effect=class_name),
        GetWindowThreadProcessId=Mock(side_effect=process),
        EnumWindows=Mock(side_effect=enumerate_windows),
    )


class LiveWindowDiscoveryTests(unittest.TestCase):
    def capture(self, hwnd=None):
        with patch("blackflow_live.capture.resolve_runtime", return_value="test-runtime"):
            return WindowsGameCapture(hwnd=hwnd)

    def test_verified_executable_accepts_title_variants_and_fullscreen_metadata(self):
        user = fake_user({123: {"title": "Arknights - CN", "pid": 42}})
        with patch("blackflow_live.capture._user32", return_value=user), \
             patch("blackflow_live.capture._process_basename", return_value="arknights.exe"), \
             patch("blackflow_live.capture.window_geometry") as read_geometry:
            result = find_game_candidates()
        self.assertEqual(result, [candidate(title="Arknights - CN")])
        read_geometry.assert_not_called()

    def test_browser_titles_qt_helpers_and_other_unity_games_are_never_targets(self):
        user = fake_user({
            1: {"title": "明日方舟", "class_name": "Chrome_WidgetWin_1", "pid": 11},
            2: {"title": "Arknights", "class_name": "Qt5152QWindowIcon", "pid": 42, "visible": False},
            3: {"title": "明日方舟", "pid": 33},
            4: {"title": "Arknights", "pid": 44},
            5: {"title": "Arknights", "pid": 55},
        })
        names = {11: "msedge.exe", 42: "arknights.exe", 33: "othergame.exe", 44: None, 55: "chrome.exe"}
        with patch("blackflow_live.capture._user32", return_value=user), \
             patch("blackflow_live.capture._process_basename", side_effect=names.get):
            self.assertEqual(find_game_candidates(), [])

    def test_minimized_and_hidden_game_clients_remain_identifiable_without_geometry(self):
        user = fake_user({123: {"minimized": True}, 124: {"visible": False}})
        with patch("blackflow_live.capture._user32", return_value=user), \
             patch("blackflow_live.capture._process_basename", return_value="arknights.exe") as process_name, \
             patch("blackflow_live.capture.window_geometry") as read_geometry:
            found = find_game_candidates()
        self.assertEqual([item.state for item in found], ["minimized", "hidden"])
        process_name.assert_called_once_with(42)
        read_geometry.assert_not_called()

    def test_discovery_rejects_hwnd_reused_while_checking_process(self):
        user = fake_user({123: {}})
        pids = iter((42, 999))
        def process(hwnd, pointer):
            pointer._obj.value = next(pids)
            return 1
        user.GetWindowThreadProcessId.side_effect = process
        with patch("blackflow_live.capture._user32", return_value=user), \
             patch("blackflow_live.capture._process_basename", return_value="arknights.exe"):
            self.assertEqual(find_game_candidates(), [])

    def test_compatibility_geometry_list_omits_unavailable_and_changed_processes(self):
        candidates = [candidate(), candidate(124, minimized=True), candidate(125, visible=False), candidate(126)]
        with patch("blackflow_live.capture.find_game_candidates", return_value=candidates), \
             patch("blackflow_live.capture.window_geometry", side_effect=[geometry(), geometry(126, 999)]) as read_geometry:
            self.assertEqual(find_game_windows(), [geometry()])
        self.assertEqual([call.args[0] for call in read_geometry.call_args_list], [123, 126])

    def test_minimized_identity_can_be_selected_but_capture_geometry_is_rejected(self):
        capture = self.capture()
        found = candidate(minimized=True)
        with patch("blackflow_live.capture.find_game_candidates", return_value=[found]), \
             patch("blackflow_live.capture.window_geometry") as read_geometry:
            self.assertEqual(capture.verified_window(), found)
            self.assertEqual(capture.hwnd, 123)
            with self.assertRaisesRegex(RuntimeError, "最小化"):
                capture.current_geometry()
        read_geometry.assert_not_called()

    def test_hidden_game_has_specific_diagnostic(self):
        capture = self.capture()
        with patch("blackflow_live.capture.find_game_candidates", return_value=[candidate(visible=False)]), \
             patch("blackflow_live.capture.window_geometry") as read_geometry:
            with self.assertRaisesRegex(RuntimeError, "隐藏"):
                capture.current_geometry()
        read_geometry.assert_not_called()

    def test_no_client_distinguishes_game_not_launched_from_process_still_loading(self):
        for processes, message in (({}, "先启动游戏客户端"), ({42: "arknights.exe"}, "检测到明日方舟进程")):
            with self.subTest(processes=processes):
                capture = self.capture()
                with patch("blackflow_live.capture.find_game_candidates", return_value=[]), \
                     patch("blackflow_live.capture._process_snapshot", return_value=processes):
                    with self.assertRaisesRegex(RuntimeError, message):
                        capture.verified_window()
                self.assertIsNone(capture.hwnd)

    def test_multiple_clients_require_selection_and_do_not_choose_implicitly(self):
        capture = self.capture()
        with patch("blackflow_live.capture.find_game_candidates", return_value=[candidate(), candidate(124, 43)]):
            with self.assertRaisesRegex(RuntimeError, "多个"):
                capture.verified_window()
        self.assertIsNone(capture.hwnd)

    def test_explicit_hwnd_cannot_bypass_game_verification(self):
        capture = self.capture(hwnd=999)
        with patch("blackflow_live.capture.find_game_candidates", return_value=[candidate()]), \
             patch("blackflow_live.capture.window_geometry") as read_geometry:
            with self.assertRaisesRegex(RuntimeError, "已关闭或身份已改变"):
                capture.current_geometry()
        read_geometry.assert_not_called()

    def test_reused_hwnd_with_different_pid_is_rejected_and_native_capture_closed(self):
        capture = self.capture()
        native = Mock()
        with patch("blackflow_live.capture.find_game_candidates", side_effect=[[candidate()], [candidate(pid=999)]]):
            capture.verified_window()
            capture._native = native
            with self.assertRaisesRegex(RuntimeError, "进程已改变"):
                capture.verified_window()
        native.close.assert_called_once_with()
        self.assertIsNone(capture._native)
        self.assertEqual(capture._selected_pid, 42)

    def test_closed_window_does_not_silently_switch_to_another_game(self):
        capture = self.capture()
        with patch("blackflow_live.capture.find_game_candidates", side_effect=[[candidate()], [candidate(124, 43)]]):
            capture.verified_window()
            with self.assertRaisesRegex(RuntimeError, "已关闭或身份已改变"):
                capture.verified_window()
        self.assertEqual(capture.hwnd, 123)

    def test_geometry_pid_race_is_rejected(self):
        capture = self.capture()
        with patch("blackflow_live.capture.find_game_candidates", return_value=[candidate()]), \
             patch("blackflow_live.capture.window_geometry", return_value=geometry(pid=999)):
            with self.assertRaisesRegex(RuntimeError, "读取期间.*进程已改变"):
                capture.current_geometry()

    def test_discovery_reports_state_and_only_attaches_usable_geometry(self):
        candidates = [candidate(), candidate(124, minimized=True), candidate(125, visible=False)]
        with patch("blackflow_live.capture.find_game_candidates", return_value=candidates), \
             patch("blackflow_live.capture.window_geometry", return_value=geometry()) as read_geometry:
            result = WindowsGameCapture.discover_windows()
        self.assertEqual([item["state"] for item in result], ["visible", "minimized", "hidden"])
        self.assertEqual(result[0]["client_rect"]["width"], 2878)
        self.assertEqual(result[0]["dpi"], 192)
        self.assertNotIn("client_rect", result[1])
        self.assertNotIn("dpi", result[2])
        read_geometry.assert_called_once_with(123)

    def test_explicit_selection_pins_minimized_identity_and_closes_previous_capture(self):
        capture = self.capture()
        native = Mock()
        capture._native = native
        found = candidate(minimized=True)
        with patch("blackflow_live.capture.find_game_candidates", return_value=[found]):
            self.assertEqual(capture.select_window(123), found)
        native.close.assert_called_once_with()
        self.assertEqual(capture._selected_pid, 42)

    def test_frame_from_other_window_cannot_pass_input_validation(self):
        capture = self.capture()
        with patch("blackflow_live.capture.find_game_candidates", return_value=[candidate()]), \
             patch("blackflow_live.capture.window_geometry") as read_geometry:
            with self.assertRaisesRegex(RuntimeError, "身份已改变"):
                capture.assert_geometry_current(geometry(hwnd=999))
        read_geometry.assert_not_called()


class ProcessNameFallbackTests(unittest.TestCase):
    def test_denied_process_handle_uses_pid_snapshot(self):
        kernel = SimpleNamespace(OpenProcess=Mock(return_value=0), QueryFullProcessImageNameW=Mock(), CloseHandle=Mock())
        with patch("blackflow_live.capture.ct.WinDLL", return_value=kernel, create=True), \
             patch("blackflow_live.capture._process_snapshot", return_value={42: "arknights.exe"}) as snapshot:
            self.assertEqual(_process_basename(42), "arknights.exe")
        snapshot.assert_called_once_with()
        kernel.QueryFullProcessImageNameW.assert_not_called()
        kernel.CloseHandle.assert_not_called()

    def test_failed_path_query_closes_handle_and_uses_snapshot(self):
        kernel = SimpleNamespace(OpenProcess=Mock(return_value=100), QueryFullProcessImageNameW=Mock(return_value=False), CloseHandle=Mock())
        with patch("blackflow_live.capture.ct.WinDLL", return_value=kernel, create=True), \
             patch("blackflow_live.capture._process_snapshot", return_value={42: "arknights.exe"}):
            self.assertEqual(_process_basename(42), "arknights.exe")
        kernel.CloseHandle.assert_called_once_with(100)

    def test_successful_path_query_uses_only_basename_and_closes_handle(self):
        def query(handle, flags, buffer, length):
            buffer.value = "D:\\Game\\Arknights.exe"
            return True
        kernel = SimpleNamespace(OpenProcess=Mock(return_value=100), QueryFullProcessImageNameW=Mock(side_effect=query), CloseHandle=Mock())
        with patch("blackflow_live.capture.ct.WinDLL", return_value=kernel, create=True), \
             patch("blackflow_live.capture._process_snapshot") as snapshot:
            self.assertEqual(_process_basename(42), "arknights.exe")
        snapshot.assert_not_called()
        kernel.CloseHandle.assert_called_once_with(100)

    def test_unidentifiable_process_is_not_assumed_to_be_a_game(self):
        kernel = SimpleNamespace(OpenProcess=Mock(return_value=0), QueryFullProcessImageNameW=Mock(), CloseHandle=Mock())
        with patch("blackflow_live.capture.ct.WinDLL", return_value=kernel, create=True), \
             patch("blackflow_live.capture._process_snapshot", return_value={43: "arknights.exe"}):
            self.assertIsNone(_process_basename(42))

    def test_process_snapshot_returns_pid_names_and_closes_handle(self):
        rows = iter(((42, "Arknights.exe"), (43, "msedge.exe")))
        def read(handle, pointer):
            row = next(rows, None)
            if row is None:
                return False
            self.assertEqual(pointer._obj.dwSize, ct.sizeof(pointer._obj))
            pointer._obj.th32ProcessID, pointer._obj.szExeFile = row
            return True
        kernel = SimpleNamespace(
            CreateToolhelp32Snapshot=Mock(return_value=200),
            Process32FirstW=Mock(side_effect=read), Process32NextW=Mock(side_effect=read), CloseHandle=Mock(),
        )
        with patch("blackflow_live.capture.ct.WinDLL", return_value=kernel, create=True):
            self.assertEqual(_process_snapshot(), {42: "arknights.exe", 43: "msedge.exe"})
        kernel.CreateToolhelp32Snapshot.assert_called_once_with(2, 0)
        kernel.CloseHandle.assert_called_once_with(200)

    def test_failed_snapshot_does_not_enumerate_or_close_invalid_handle(self):
        kernel = SimpleNamespace(
            CreateToolhelp32Snapshot=Mock(return_value=ct.c_void_p(-1).value),
            Process32FirstW=Mock(), Process32NextW=Mock(), CloseHandle=Mock(),
        )
        with patch("blackflow_live.capture.ct.WinDLL", return_value=kernel, create=True):
            self.assertEqual(_process_snapshot(), {})
        kernel.Process32FirstW.assert_not_called()
        kernel.CloseHandle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
