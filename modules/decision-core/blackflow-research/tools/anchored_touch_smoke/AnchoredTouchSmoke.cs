using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace AnchoredTouchSmoke
{
    public static class Program
    {
        private const uint WS_OVERLAPPEDWINDOW = 0x00CF0000;
        private const uint WS_VISIBLE = 0x10000000;
        private const uint WM_DESTROY = 0x0002;
        private const uint WM_PAINT = 0x000F;
        private const uint WM_LBUTTONDOWN = 0x0201;
        private const uint WM_LBUTTONUP = 0x0202;
        private const uint WM_POINTERUPDATE = 0x0245;
        private const uint WM_POINTERDOWN = 0x0246;
        private const uint WM_POINTERUP = 0x0247;
        private const uint WM_APP_DONE = 0x8001;
        private const int SW_SHOWNORMAL = 1;
        private const uint SWP_NOSIZE = 0x0001;
        private const uint SWP_NOMOVE = 0x0002;
        private const uint SWP_NOACTIVATE = 0x0010;
        private const uint PT_TOUCH = 0x00000002;
        private const uint LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;

        private const int ButtonLeft = 100;
        private const int ButtonTop = 100;
        private const int ButtonRight = 300;
        private const int ButtonBottom = 170;
        private const int ButtonX = 200;
        private const int ButtonY = 135;

        private static readonly WndProcDelegate WndProcRoot = WindowProc;
        private static readonly ConcurrentQueue<string> PointerEventLog = new ConcurrentQueue<string>();

        private static IntPtr _targetWindow;
        private static IntPtr _sentinelWindow;
        private static int _pointerDownCount;
        private static int _pointerUpCount;
        private static int _pointerUpdateCount;
        private static int _mouseDownCount;
        private static int _mouseUpCount;
        private static int _buttonActivationCount;
        private static string _resultText = string.Empty;
        private static Exception _workerError;

        public static string Run(string controlDllPath)
        {
            if (string.IsNullOrWhiteSpace(controlDllPath))
            {
                throw new ArgumentException("controlDllPath is required", nameof(controlDllPath));
            }

            controlDllPath = System.IO.Path.GetFullPath(controlDllPath);
            if (!System.IO.File.Exists(controlDllPath))
            {
                throw new System.IO.FileNotFoundException("MaaWin32ControlUnit.dll was not found", controlDllPath);
            }

            Native.SetProcessDpiAwarenessContext(new IntPtr(-4));

            string suffix = Native.GetCurrentProcessId().ToString();
            string targetClass = "MaaAnchoredTouchSmokeTarget_" + suffix;
            string sentinelClass = "MaaAnchoredTouchSmokeSentinel_" + suffix;
            RegisterWindowClass(targetClass);
            RegisterWindowClass(sentinelClass);

            _targetWindow = Native.CreateWindowExW(
                0,
                targetClass,
                "AnchoredTouch harmless target",
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                50,
                60,
                440,
                320,
                IntPtr.Zero,
                IntPtr.Zero,
                Native.GetModuleHandleW(null),
                IntPtr.Zero);
            EnsureHandle(_targetWindow, "CreateWindowEx(target)");

            _sentinelWindow = Native.CreateWindowExW(
                0,
                sentinelClass,
                "Foreground sentinel - should stay active",
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                560,
                80,
                380,
                220,
                IntPtr.Zero,
                IntPtr.Zero,
                Native.GetModuleHandleW(null),
                IntPtr.Zero);
            EnsureHandle(_sentinelWindow, "CreateWindowEx(sentinel)");

            Native.ShowWindow(_targetWindow, SW_SHOWNORMAL);
            Native.ShowWindow(_sentinelWindow, SW_SHOWNORMAL);
            Native.UpdateWindow(_targetWindow);
            Native.UpdateWindow(_sentinelWindow);
            bool pointerTargetRegistered = Native.RegisterPointerInputTarget(_targetWindow, PT_TOUCH);

            // Put the target near the front without activating it, then ask Windows to
            // foreground a separate sentinel. Foreground-lock policy may reject that
            // request; either way, the target itself must never become foreground.
            Native.SetWindowPos(_targetWindow, IntPtr.Zero, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            Native.SetForegroundWindow(_sentinelWindow);

            Thread worker = new Thread(() => RunWorker(controlDllPath, pointerTargetRegistered));
            worker.IsBackground = true;
            worker.Name = "AnchoredTouchSmokeWorker";
            worker.Start();

            MSG message;
            while (Native.GetMessageW(out message, IntPtr.Zero, 0, 0) > 0)
            {
                Native.TranslateMessage(ref message);
                Native.DispatchMessageW(ref message);
            }

            worker.Join();
            if (_workerError != null)
            {
                throw new InvalidOperationException("AnchoredTouch smoke worker failed", _workerError);
            }

            Console.WriteLine(_resultText);
            return _resultText;
        }

        private static void RunWorker(string controlDllPath, bool pointerTargetRegistered)
        {
            IntPtr module = IntPtr.Zero;
            IntPtr unit = IntPtr.Zero;
            try
            {
                Thread.Sleep(450);

                string dependencyDirectory = System.IO.Path.GetDirectoryName(controlDllPath);
                Native.SetDllDirectoryW(dependencyDirectory);
                module = Native.LoadLibraryExW(controlDllPath, IntPtr.Zero, LOAD_WITH_ALTERED_SEARCH_PATH);
                EnsureHandle(module, "LoadLibraryEx(MaaWin32ControlUnit)");

                GetVersionDelegate getVersion = GetExport<GetVersionDelegate>(module, "MaaWin32ControlUnitGetVersion");
                CreateDelegate create = GetExport<CreateDelegate>(module, "MaaWin32ControlUnitCreate");
                DestroyDelegate destroy = GetExport<DestroyDelegate>(module, "MaaWin32ControlUnitDestroy");
                string version = Marshal.PtrToStringAnsi(getVersion()) ?? string.Empty;

                // GDI screencap = 1, AnchoredTouch mouse = 1024, SendMessage keyboard = 2.
                unit = create(_targetWindow, 1UL, 1024UL, 2UL);
                EnsureHandle(unit, "MaaWin32ControlUnitCreate");

                IntPtr vtable = Marshal.ReadIntPtr(unit);
                EnsureHandle(vtable, "ControlUnitAPI vtable");

                BoolThisDelegate connect = GetVtableDelegate<BoolThisDelegate>(vtable, 1);
                BoolThisDelegate connected = GetVtableDelegate<BoolThisDelegate>(vtable, 2);
                GetFeaturesDelegate getFeatures = GetVtableDelegate<GetFeaturesDelegate>(vtable, 4);
                ScreencapDelegate screencap = GetVtableDelegate<ScreencapDelegate>(vtable, 7);
                TouchDownDelegate touchDown = GetVtableDelegate<TouchDownDelegate>(vtable, 10);
                TouchUpDelegate touchUp = GetVtableDelegate<TouchUpDelegate>(vtable, 12);

                bool connectOk = connect(unit);
                bool connectedOk = connected(unit);
                ulong features = getFeatures(unit);

                IntPtr opencv = Native.GetModuleHandleW("opencv_world4_maa.dll");
                EnsureHandle(opencv, "GetModuleHandle(opencv_world4_maa.dll)");
                MatCtorDelegate matCtor = GetExport<MatCtorDelegate>(opencv, "??0Mat@cv@@QEAA@XZ");
                MatDtorDelegate matDtor = GetExport<MatDtorDelegate>(opencv, "??1Mat@cv@@QEAA@XZ");

                bool screencapOk;
                int imageRows;
                int imageCols;
                IntPtr mat = Marshal.AllocHGlobal(128);
                try
                {
                    Marshal.Copy(new byte[128], 0, mat, 128);
                    matCtor(mat);
                    try
                    {
                        screencapOk = screencap(unit, mat);
                        imageRows = Marshal.ReadInt32(mat, 8);
                        imageCols = Marshal.ReadInt32(mat, 12);
                    }
                    finally
                    {
                        matDtor(mat);
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(mat);
                }

                // Establish the exact pre-touch baseline after connect and screencap.
                Native.SetWindowPos(_targetWindow, IntPtr.Zero, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
                Native.SetForegroundWindow(_sentinelWindow);
                Thread.Sleep(200);

                POINT baselineCursor;
                RECT baselineRect;
                Native.GetCursorPos(out baselineCursor);
                Native.GetWindowRect(_targetWindow, out baselineRect);
                IntPtr baselineForeground = Native.GetForegroundWindow();

                Sampler sampler = new Sampler(_targetWindow, baselineCursor, baselineRect, baselineForeground);
                sampler.Start();
                Thread.Sleep(60);

                long touchStartTicks = DateTime.UtcNow.Ticks;
                bool touchDownOk = touchDown(unit, 0, ButtonX, ButtonY, 0);
                Thread.Sleep(80);
                bool touchUpOk = touchUp(unit, 0);
                long touchEndTicks = DateTime.UtcNow.Ticks;

                Thread.Sleep(300);
                sampler.Stop();

                POINT postCursor;
                RECT postRect;
                Native.GetCursorPos(out postCursor);
                Native.GetWindowRect(_targetWindow, out postRect);
                IntPtr postForeground = Native.GetForegroundWindow();

                int pointerDown = Volatile.Read(ref _pointerDownCount);
                int pointerUp = Volatile.Read(ref _pointerUpCount);
                int pointerUpdate = Volatile.Read(ref _pointerUpdateCount);
                int mouseDown = Volatile.Read(ref _mouseDownCount);
                int mouseUp = Volatile.Read(ref _mouseUpCount);
                int activations = Volatile.Read(ref _buttonActivationCount);

                bool cursorInvariant = sampler.CursorChangeCount == 0 && baselineCursor.Equals(postCursor);
                bool rectInvariant = sampler.RectChangeCount == 0 && baselineRect.Equals(postRect);
                bool foregroundInvariant = sampler.ForegroundChangeCount == 0
                    && postForeground == baselineForeground
                    && baselineForeground != _targetWindow;
                bool inputDelivered = touchDownOk && touchUpOk && pointerDown > 0 && pointerUp > 0 && activations > 0;
                bool passed = connectOk && connectedOk && screencapOk && imageRows > 0 && imageCols > 0
                    && inputDelivered && cursorInvariant && rectInvariant && foregroundInvariant;

                StringBuilder events = new StringBuilder();
                string eventItem;
                while (PointerEventLog.TryDequeue(out eventItem))
                {
                    if (events.Length > 0)
                    {
                        events.Append(" | ");
                    }
                    events.Append(eventItem);
                }

                StringBuilder output = new StringBuilder();
                output.AppendLine("ANCHOR_TOUCH_SMOKE_RESULT");
                output.AppendLine("pass=" + passed);
                output.AppendLine("dll_version=" + version);
                output.AppendLine("direct_vtable_connect=" + connectOk);
                output.AppendLine("direct_vtable_connected=" + connectedOk);
                output.AppendLine("features=" + features);
                output.AppendLine("direct_vtable_screencap=" + screencapOk);
                output.AppendLine("screencap_size=" + imageCols + "x" + imageRows);
                output.AppendLine("pointer_target_registered=" + pointerTargetRegistered);
                output.AppendLine("direct_vtable_touch_down=" + touchDownOk);
                output.AppendLine("direct_vtable_touch_up=" + touchUpOk);
                output.AppendLine("touch_duration_ms=" + TimeSpan.FromTicks(touchEndTicks - touchStartTicks).TotalMilliseconds.ToString("F3"));
                output.AppendLine("pointer_down=" + pointerDown);
                output.AppendLine("pointer_update=" + pointerUpdate);
                output.AppendLine("pointer_up=" + pointerUp);
                output.AppendLine("button_activations=" + activations);
                output.AppendLine("promoted_mouse_down=" + mouseDown);
                output.AppendLine("promoted_mouse_up=" + mouseUp);
                output.AppendLine("sample_count=" + sampler.SampleCount);
                output.AppendLine("cursor_invariant=" + cursorInvariant);
                output.AppendLine("cursor_baseline=" + baselineCursor);
                output.AppendLine("cursor_post=" + postCursor);
                output.AppendLine("cursor_change_samples=" + sampler.CursorChangeCount);
                output.AppendLine("window_rect_invariant=" + rectInvariant);
                output.AppendLine("window_rect_baseline=" + baselineRect);
                output.AppendLine("window_rect_post=" + postRect);
                output.AppendLine("window_rect_change_samples=" + sampler.RectChangeCount);
                output.AppendLine("foreground_invariant=" + foregroundInvariant);
                output.AppendLine("foreground_baseline=0x" + baselineForeground.ToInt64().ToString("X"));
                output.AppendLine("foreground_post=0x" + postForeground.ToInt64().ToString("X"));
                output.AppendLine("target_hwnd=0x" + _targetWindow.ToInt64().ToString("X"));
                output.AppendLine("sentinel_hwnd=0x" + _sentinelWindow.ToInt64().ToString("X"));
                output.AppendLine("foreground_change_samples=" + sampler.ForegroundChangeCount);
                output.AppendLine("pointer_events=" + events);
                _resultText = output.ToString().TrimEnd();

                destroy(unit);
                unit = IntPtr.Zero;
            }
            catch (Exception ex)
            {
                _workerError = ex;
            }
            finally
            {
                if (unit != IntPtr.Zero && module != IntPtr.Zero)
                {
                    try
                    {
                        GetExport<DestroyDelegate>(module, "MaaWin32ControlUnitDestroy")(unit);
                    }
                    catch
                    {
                    }
                }
                if (module != IntPtr.Zero)
                {
                    Native.FreeLibrary(module);
                }
                Native.PostMessageW(_targetWindow, WM_APP_DONE, IntPtr.Zero, IntPtr.Zero);
            }
        }

        private static void RegisterWindowClass(string className)
        {
            WNDCLASSEX wc = new WNDCLASSEX();
            wc.cbSize = (uint)Marshal.SizeOf(typeof(WNDCLASSEX));
            wc.lpfnWndProc = Marshal.GetFunctionPointerForDelegate(WndProcRoot);
            wc.hInstance = Native.GetModuleHandleW(null);
            wc.hCursor = Native.LoadCursorW(IntPtr.Zero, new IntPtr(32512));
            wc.hbrBackground = new IntPtr(6); // COLOR_WINDOW + 1
            wc.lpszClassName = className;
            if (Native.RegisterClassExW(ref wc) == 0)
            {
                throw new InvalidOperationException("RegisterClassEx failed: " + Marshal.GetLastWin32Error());
            }
        }

        private static IntPtr WindowProc(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam)
        {
            if (hwnd == _targetWindow)
            {
                if (msg == WM_POINTERDOWN || msg == WM_POINTERUPDATE || msg == WM_POINTERUP)
                {
                    POINT point = PointFromLParam(lParam);
                    POINT clientPoint = point;
                    Native.ScreenToClient(hwnd, ref clientPoint);
                    uint pointerId = unchecked((uint)wParam.ToInt64()) & 0xFFFF;
                    PointerEventLog.Enqueue(
                        "0x" + msg.ToString("X") + " id=" + pointerId + " client=" + clientPoint);

                    if (msg == WM_POINTERDOWN)
                    {
                        Interlocked.Increment(ref _pointerDownCount);
                    }
                    else if (msg == WM_POINTERUPDATE)
                    {
                        Interlocked.Increment(ref _pointerUpdateCount);
                    }
                    else
                    {
                        Interlocked.Increment(ref _pointerUpCount);
                        if (clientPoint.X >= ButtonLeft && clientPoint.X < ButtonRight
                            && clientPoint.Y >= ButtonTop && clientPoint.Y < ButtonBottom)
                        {
                            Interlocked.Increment(ref _buttonActivationCount);
                            Native.InvalidateRect(hwnd, IntPtr.Zero, true);
                        }
                    }
                }
                else if (msg == WM_LBUTTONDOWN)
                {
                    Interlocked.Increment(ref _mouseDownCount);
                }
                else if (msg == WM_LBUTTONUP)
                {
                    Interlocked.Increment(ref _mouseUpCount);
                }
                else if (msg == WM_PAINT)
                {
                    DrawTargetWindow(hwnd);
                    return IntPtr.Zero;
                }
                else if (msg == WM_APP_DONE)
                {
                    if (_sentinelWindow != IntPtr.Zero)
                    {
                        Native.DestroyWindow(_sentinelWindow);
                        _sentinelWindow = IntPtr.Zero;
                    }
                    Native.DestroyWindow(hwnd);
                    _targetWindow = IntPtr.Zero;
                    Native.PostQuitMessage(0);
                    return IntPtr.Zero;
                }
            }
            else if (hwnd == _sentinelWindow && msg == WM_PAINT)
            {
                DrawSentinelWindow(hwnd);
                return IntPtr.Zero;
            }

            if (msg == WM_DESTROY)
            {
                return IntPtr.Zero;
            }
            return Native.DefWindowProcW(hwnd, msg, wParam, lParam);
        }

        private static void DrawTargetWindow(IntPtr hwnd)
        {
            PAINTSTRUCT ps;
            IntPtr dc = Native.BeginPaint(hwnd, out ps);
            try
            {
                RECT button = new RECT(ButtonLeft, ButtonTop, ButtonRight, ButtonBottom);
                IntPtr brush = Native.CreateSolidBrush(Volatile.Read(ref _buttonActivationCount) > 0 ? 0x00A0E0A0u : 0x00E0A060u);
                Native.FillRect(dc, ref button, brush);
                Native.DeleteObject(brush);
                Native.SetBkMode(dc, 1);
                Native.SetTextColor(dc, 0x00000000);
                string text = Volatile.Read(ref _buttonActivationCount) > 0
                    ? "Synthetic touch received"
                    : "Touch target";
                Native.DrawTextW(dc, text, text.Length, ref button, 0x00000001 | 0x00000004 | 0x00000020);
            }
            finally
            {
                Native.EndPaint(hwnd, ref ps);
            }
        }

        private static void DrawSentinelWindow(IntPtr hwnd)
        {
            PAINTSTRUCT ps;
            IntPtr dc = Native.BeginPaint(hwnd, out ps);
            try
            {
                RECT rect;
                Native.GetClientRect(hwnd, out rect);
                Native.SetBkMode(dc, 1);
                Native.SetTextColor(dc, 0x00000000);
                string text = "This window must remain foreground";
                Native.DrawTextW(dc, text, text.Length, ref rect, 0x00000001 | 0x00000004 | 0x00000020);
            }
            finally
            {
                Native.EndPaint(hwnd, ref ps);
            }
        }

        private static POINT PointFromLParam(IntPtr lParam)
        {
            long value = lParam.ToInt64();
            return new POINT(unchecked((short)(value & 0xFFFF)), unchecked((short)((value >> 16) & 0xFFFF)));
        }

        private static T GetExport<T>(IntPtr module, string name) where T : class
        {
            IntPtr address = Native.GetProcAddress(module, name);
            EnsureHandle(address, "GetProcAddress(" + name + ")");
            return (T)(object)Marshal.GetDelegateForFunctionPointer(address, typeof(T));
        }

        private static T GetVtableDelegate<T>(IntPtr vtable, int slot) where T : class
        {
            IntPtr address = Marshal.ReadIntPtr(vtable, slot * IntPtr.Size);
            EnsureHandle(address, "ControlUnitAPI vtable slot " + slot);
            return (T)(object)Marshal.GetDelegateForFunctionPointer(address, typeof(T));
        }

        private static void EnsureHandle(IntPtr handle, string operation)
        {
            if (handle == IntPtr.Zero)
            {
                throw new InvalidOperationException(operation + " failed; Win32 error=" + Marshal.GetLastWin32Error());
            }
        }

        private sealed class Sampler
        {
            private readonly IntPtr _target;
            private readonly POINT _baselineCursor;
            private readonly RECT _baselineRect;
            private readonly IntPtr _baselineForeground;
            private readonly Thread _thread;
            private volatile bool _stop;

            public Sampler(IntPtr target, POINT baselineCursor, RECT baselineRect, IntPtr baselineForeground)
            {
                _target = target;
                _baselineCursor = baselineCursor;
                _baselineRect = baselineRect;
                _baselineForeground = baselineForeground;
                _thread = new Thread(SampleLoop);
                _thread.IsBackground = true;
                _thread.Name = "AnchoredTouchInvariantSampler";
            }

            public int SampleCount;
            public int CursorChangeCount;
            public int RectChangeCount;
            public int ForegroundChangeCount;

            public void Start()
            {
                _thread.Start();
            }

            public void Stop()
            {
                _stop = true;
                _thread.Join();
            }

            private void SampleLoop()
            {
                while (!_stop)
                {
                    POINT cursor;
                    RECT rect;
                    Native.GetCursorPos(out cursor);
                    Native.GetWindowRect(_target, out rect);
                    IntPtr foreground = Native.GetForegroundWindow();
                    Interlocked.Increment(ref SampleCount);
                    if (!cursor.Equals(_baselineCursor))
                    {
                        Interlocked.Increment(ref CursorChangeCount);
                    }
                    if (!rect.Equals(_baselineRect))
                    {
                        Interlocked.Increment(ref RectChangeCount);
                    }
                    if (foreground != _baselineForeground)
                    {
                        Interlocked.Increment(ref ForegroundChangeCount);
                    }
                    Thread.SpinWait(64);
                }
            }
        }

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        private delegate IntPtr WndProcDelegate(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr GetVersionDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr CreateDelegate(IntPtr hwnd, ulong screencapMethod, ulong mouseMethod, ulong keyboardMethod);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void DestroyDelegate(IntPtr unit);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private delegate bool BoolThisDelegate(IntPtr unit);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate ulong GetFeaturesDelegate(IntPtr unit);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private delegate bool ScreencapDelegate(IntPtr unit, IntPtr mat);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private delegate bool TouchDownDelegate(IntPtr unit, int contact, int x, int y, int pressure);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private delegate bool TouchUpDelegate(IntPtr unit, int contact);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void MatCtorDelegate(IntPtr mat);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void MatDtorDelegate(IntPtr mat);

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct WNDCLASSEX
        {
            public uint cbSize;
            public uint style;
            public IntPtr lpfnWndProc;
            public int cbClsExtra;
            public int cbWndExtra;
            public IntPtr hInstance;
            public IntPtr hIcon;
            public IntPtr hCursor;
            public IntPtr hbrBackground;
            public string lpszMenuName;
            public string lpszClassName;
            public IntPtr hIconSm;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct MSG
        {
            public IntPtr hwnd;
            public uint message;
            public IntPtr wParam;
            public IntPtr lParam;
            public uint time;
            public POINT pt;
            public uint lPrivate;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PAINTSTRUCT
        {
            public IntPtr hdc;
            public int fErase;
            public RECT rcPaint;
            public int fRestore;
            public int fIncUpdate;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
            public byte[] rgbReserved;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct POINT : IEquatable<POINT>
        {
            public int X;
            public int Y;

            public POINT(int x, int y)
            {
                X = x;
                Y = y;
            }

            public bool Equals(POINT other)
            {
                return X == other.X && Y == other.Y;
            }

            public override string ToString()
            {
                return X + "," + Y;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct RECT : IEquatable<RECT>
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;

            public RECT(int left, int top, int right, int bottom)
            {
                Left = left;
                Top = top;
                Right = right;
                Bottom = bottom;
            }

            public bool Equals(RECT other)
            {
                return Left == other.Left && Top == other.Top && Right == other.Right && Bottom == other.Bottom;
            }

            public override string ToString()
            {
                return Left + "," + Top + "," + Right + "," + Bottom;
            }
        }

        private static class Native
        {
            [DllImport("kernel32.dll")]
            public static extern uint GetCurrentProcessId();

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
            public static extern IntPtr GetModuleHandleW(string moduleName);

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            public static extern bool SetDllDirectoryW(string path);

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            public static extern IntPtr LoadLibraryExW(string path, IntPtr file, uint flags);

            [DllImport("kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
            public static extern IntPtr GetProcAddress(IntPtr module, string name);

            [DllImport("kernel32.dll", SetLastError = true)]
            public static extern bool FreeLibrary(IntPtr module);

            [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            public static extern ushort RegisterClassExW(ref WNDCLASSEX windowClass);

            [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            public static extern IntPtr CreateWindowExW(
                uint exStyle,
                string className,
                string windowName,
                uint style,
                int x,
                int y,
                int width,
                int height,
                IntPtr parent,
                IntPtr menu,
                IntPtr instance,
                IntPtr param);

            [DllImport("user32.dll")]
            public static extern bool DestroyWindow(IntPtr hwnd);

            [DllImport("user32.dll")]
            public static extern bool ShowWindow(IntPtr hwnd, int command);

            [DllImport("user32.dll")]
            public static extern bool UpdateWindow(IntPtr hwnd);

            [DllImport("user32.dll")]
            public static extern bool SetForegroundWindow(IntPtr hwnd);

            [DllImport("user32.dll")]
            public static extern IntPtr GetForegroundWindow();

            [DllImport("user32.dll")]
            public static extern bool SetWindowPos(IntPtr hwnd, IntPtr insertAfter, int x, int y, int cx, int cy, uint flags);

            [DllImport("user32.dll")]
            public static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);

            [DllImport("user32.dll")]
            public static extern bool GetClientRect(IntPtr hwnd, out RECT rect);

            [DllImport("user32.dll")]
            public static extern bool GetCursorPos(out POINT point);

            [DllImport("user32.dll")]
            public static extern bool ScreenToClient(IntPtr hwnd, ref POINT point);

            [DllImport("user32.dll")]
            public static extern bool RegisterPointerInputTarget(IntPtr hwnd, uint pointerType);

            [DllImport("user32.dll")]
            public static extern bool SetProcessDpiAwarenessContext(IntPtr context);

            [DllImport("user32.dll", CharSet = CharSet.Unicode)]
            public static extern IntPtr LoadCursorW(IntPtr instance, IntPtr cursorName);

            [DllImport("user32.dll")]
            public static extern IntPtr DefWindowProcW(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);

            [DllImport("user32.dll")]
            public static extern int GetMessageW(out MSG message, IntPtr hwnd, uint min, uint max);

            [DllImport("user32.dll")]
            public static extern bool TranslateMessage(ref MSG message);

            [DllImport("user32.dll")]
            public static extern IntPtr DispatchMessageW(ref MSG message);

            [DllImport("user32.dll")]
            public static extern bool PostMessageW(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);

            [DllImport("user32.dll")]
            public static extern void PostQuitMessage(int exitCode);

            [DllImport("user32.dll")]
            public static extern bool InvalidateRect(IntPtr hwnd, IntPtr rect, bool erase);

            [DllImport("user32.dll")]
            public static extern IntPtr BeginPaint(IntPtr hwnd, out PAINTSTRUCT paint);

            [DllImport("user32.dll")]
            public static extern bool EndPaint(IntPtr hwnd, ref PAINTSTRUCT paint);

            [DllImport("user32.dll")]
            public static extern int FillRect(IntPtr dc, ref RECT rect, IntPtr brush);

            [DllImport("user32.dll", CharSet = CharSet.Unicode)]
            public static extern int DrawTextW(IntPtr dc, string text, int count, ref RECT rect, uint format);

            [DllImport("gdi32.dll")]
            public static extern IntPtr CreateSolidBrush(uint color);

            [DllImport("gdi32.dll")]
            public static extern bool DeleteObject(IntPtr obj);

            [DllImport("gdi32.dll")]
            public static extern int SetBkMode(IntPtr dc, int mode);

            [DllImport("gdi32.dll")]
            public static extern uint SetTextColor(IntPtr dc, uint color);
        }
    }
}
