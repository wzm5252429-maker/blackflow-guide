param(
    [string]$MaaFrameworkBin = "$env:LOCALAPPDATA\Temp\maafw-v5.13.0-beta.5-x64\bin",
    [switch]$KeepWindowOpen
)

$ErrorActionPreference = 'Stop'

$source = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

public sealed class AnchoredTouchTestWindow : IDisposable
{
    private const uint WM_DESTROY = 0x0002;
    private const uint WM_LBUTTONDOWN = 0x0201;
    private const uint WM_LBUTTONUP = 0x0202;
    private const uint WM_POINTERUPDATE = 0x0245;
    private const uint WM_POINTERDOWN = 0x0246;
    private const uint WM_POINTERUP = 0x0247;
    private const uint WS_OVERLAPPEDWINDOW = 0x00CF0000;
    private const uint WS_VISIBLE = 0x10000000;
    private const uint WS_EX_NOACTIVATE = 0x08000000;
    private const uint WS_EX_TOPMOST = 0x00000008;
    private const int SW_SHOWNOACTIVATE = 4;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct WNDCLASSEX
    {
        public uint cbSize;
        public uint style;
        public WndProc lpfnWndProc;
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
    public struct RECT
    {
        public int Left, Top, Right, Bottom;
        public override string ToString() => $"({Left},{Top})-({Right},{Bottom})";
        public override bool Equals(object obj)
        {
            if (!(obj is RECT)) return false;
            var other = (RECT)obj;
            return Left == other.Left && Top == other.Top && Right == other.Right && Bottom == other.Bottom;
        }
        public override int GetHashCode() => Left ^ Top ^ Right ^ Bottom;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct POINT
    {
        public int X, Y;
        public override string ToString() => $"({X},{Y})";
        public override bool Equals(object obj)
        {
            if (!(obj is POINT)) return false;
            var other = (POINT)obj;
            return X == other.X && Y == other.Y;
        }
        public override int GetHashCode() => X ^ Y;
    }

    private delegate IntPtr WndProc(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
    private readonly ManualResetEventSlim ready = new ManualResetEventSlim(false);
    private readonly List<string> messages = new List<string>();
    private Thread thread;
    private WndProc wndProc;
    private string className;
    private IntPtr hwnd;

    public IntPtr Handle => hwnd;
    public string[] Messages { get { lock (messages) return messages.ToArray(); } }
    public int PointerDownCount { get { lock (messages) return messages.FindAll(x => x.StartsWith("WM_POINTERDOWN")).Count; } }
    public int PointerUpCount { get { lock (messages) return messages.FindAll(x => x.StartsWith("WM_POINTERUP ")).Count; } }

    public void Start()
    {
        thread = new Thread(WindowThread) { IsBackground = true, Name = "AnchoredTouchTestWindow" };
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        if (!ready.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException("Test window did not start.");
        if (hwnd == IntPtr.Zero) throw new InvalidOperationException("CreateWindowEx failed.");
    }

    private void WindowThread()
    {
        SetThreadDpiAwarenessContext((IntPtr)(-4)); // PER_MONITOR_AWARE_V2, matching Maa's injection thread
        wndProc = WindowProc;
        className = "CodexAnchoredTouchProbe_" + Guid.NewGuid().ToString("N");
        var wc = new WNDCLASSEX
        {
            cbSize = (uint)Marshal.SizeOf(typeof(WNDCLASSEX)),
            lpfnWndProc = wndProc,
            hInstance = GetModuleHandle(null),
            hCursor = LoadCursor(IntPtr.Zero, (IntPtr)32512),
            hbrBackground = (IntPtr)6,
            lpszClassName = className
        };
        if (RegisterClassEx(ref wc) == 0) { ready.Set(); return; }
        hwnd = CreateWindowEx(
            WS_EX_NOACTIVATE | WS_EX_TOPMOST, className, "MAA AnchoredTouch isolated probe",
            WS_OVERLAPPEDWINDOW | WS_VISIBLE, 120, 120, 640, 360,
            IntPtr.Zero, IntPtr.Zero, wc.hInstance, IntPtr.Zero);
        if (hwnd != IntPtr.Zero)
        {
            // Explicitly request PT_TOUCH pointer delivery. Real applications such as
            // Unity register their own input target; the synthetic probe must do so too.
            RegisterPointerInputTarget(hwnd, 2);
            ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        }
        ready.Set();

        MSG msg;
        while (GetMessage(out msg, IntPtr.Zero, 0, 0) > 0)
        {
            TranslateMessage(ref msg);
            DispatchMessage(ref msg);
        }
        UnregisterClass(className, wc.hInstance);
    }

    private IntPtr WindowProc(IntPtr window, uint msg, IntPtr wParam, IntPtr lParam)
    {
        if (msg == WM_POINTERDOWN || msg == WM_POINTERUP || msg == WM_POINTERUPDATE ||
            msg == WM_LBUTTONDOWN || msg == WM_LBUTTONUP)
        {
            int packed = unchecked((int)lParam.ToInt64());
            short x = unchecked((short)(packed & 0xffff));
            short y = unchecked((short)((packed >> 16) & 0xffff));
            string name = msg == WM_POINTERDOWN ? "WM_POINTERDOWN" :
                          msg == WM_POINTERUP ? "WM_POINTERUP" :
                          msg == WM_POINTERUPDATE ? "WM_POINTERUPDATE" :
                          msg == WM_LBUTTONDOWN ? "WM_LBUTTONDOWN" : "WM_LBUTTONUP";
            lock (messages) messages.Add($"{name} id={unchecked((ushort)wParam.ToInt64())} pos=({x},{y})");
        }
        if (msg == WM_DESTROY)
        {
            PostQuitMessage(0);
            return IntPtr.Zero;
        }
        return DefWindowProc(window, msg, wParam, lParam);
    }

    public RECT GetRect()
    {
        IntPtr previous = SetThreadDpiAwarenessContext((IntPtr)(-4));
        try
        {
            RECT rect;
            if (!GetWindowRect(hwnd, out rect)) throw new InvalidOperationException("GetWindowRect failed.");
            return rect;
        }
        finally { if (previous != IntPtr.Zero) SetThreadDpiAwarenessContext(previous); }
    }

    public static POINT GetCursor()
    {
        POINT point;
        if (!GetCursorPos(out point)) throw new InvalidOperationException("GetCursorPos failed.");
        return point;
    }

    public static IntPtr ForegroundWindow() => GetForegroundWindow();

    public string HitTestClientPoint(int x, int y)
    {
        IntPtr previous = SetThreadDpiAwarenessContext((IntPtr)(-4));
        try
        {
            var point = new POINT { X = x, Y = y };
            if (!ClientToScreen(hwnd, ref point)) throw new InvalidOperationException("ClientToScreen failed.");
            IntPtr hit = WindowFromPoint(point);
            return $"screen={point} hit=0x{hit.ToInt64():X} target=0x{hwnd.ToInt64():X}";
        }
        finally { if (previous != IntPtr.Zero) SetThreadDpiAwarenessContext(previous); }
    }

    public void Dispose()
    {
        if (hwnd != IntPtr.Zero) PostMessage(hwnd, 0x0010, IntPtr.Zero, IntPtr.Zero);
        if (thread != null && thread.IsAlive) thread.Join(3000);
        ready.Dispose();
    }

    [StructLayout(LayoutKind.Sequential)] private struct MSG { public IntPtr hwnd; public uint message; public UIntPtr wParam; public IntPtr lParam; public uint time; public POINT pt; public uint lPrivate; }
    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern ushort RegisterClassEx(ref WNDCLASSEX value);
    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr CreateWindowEx(uint exStyle, string className, string title, uint style, int x, int y, int width, int height, IntPtr parent, IntPtr menu, IntPtr instance, IntPtr param);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hwnd, int command);
    [DllImport("user32.dll")] private static extern sbyte GetMessage(out MSG msg, IntPtr hwnd, uint min, uint max);
    [DllImport("user32.dll")] private static extern bool TranslateMessage(ref MSG msg);
    [DllImport("user32.dll")] private static extern IntPtr DispatchMessage(ref MSG msg);
    [DllImport("user32.dll")] private static extern IntPtr DefWindowProc(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] private static extern void PostQuitMessage(int exitCode);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern bool UnregisterClass(string className, IntPtr instance);
    [DllImport("user32.dll")] private static extern IntPtr LoadCursor(IntPtr instance, IntPtr cursorName);
    [DllImport("user32.dll")] private static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);
    [DllImport("user32.dll")] private static extern bool GetCursorPos(out POINT point);
    [DllImport("user32.dll")] private static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] private static extern bool ClientToScreen(IntPtr hwnd, ref POINT point);
    [DllImport("user32.dll")] private static extern IntPtr WindowFromPoint(POINT point);
    [DllImport("user32.dll")] private static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
    [DllImport("user32.dll", SetLastError = true)] private static extern bool RegisterPointerInputTarget(IntPtr hwnd, uint pointerType);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] private static extern IntPtr GetModuleHandle(string moduleName);
}

public sealed class MaaFrameworkApi : IDisposable
{
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate IntPtr CreateDelegate(IntPtr hwnd, ulong screencap, ulong mouse, ulong keyboard);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate void DestroyDelegate(IntPtr controller);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate long PostConnectionDelegate(IntPtr controller);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate long PostScreencapDelegate(IntPtr controller);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate long PostTouchDownDelegate(IntPtr controller, int contact, int x, int y, int pressure);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate long PostTouchUpDelegate(IntPtr controller, int contact);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int WaitDelegate(IntPtr controller, long id);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate byte ConnectedDelegate(IntPtr controller);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate byte SetOptionDelegate(IntPtr controller, int key, IntPtr value, ulong valueSize);

    private IntPtr module;
    private CreateDelegate create;
    private DestroyDelegate destroy;
    private PostConnectionDelegate postConnection;
    private PostScreencapDelegate postScreencap;
    private PostTouchDownDelegate postTouchDown;
    private PostTouchUpDelegate postTouchUp;
    private WaitDelegate wait;
    private ConnectedDelegate connected;
    private SetOptionDelegate setOption;

    public MaaFrameworkApi(string binDirectory)
    {
        string fullPath = System.IO.Path.Combine(binDirectory, "MaaFramework.dll");
        SetDllDirectory(binDirectory);
        try { module = LoadLibrary(fullPath); }
        finally { SetDllDirectory(null); }
        if (module == IntPtr.Zero) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "LoadLibrary MaaFramework.dll failed");
        create = Bind<CreateDelegate>("MaaWin32ControllerCreate");
        destroy = Bind<DestroyDelegate>("MaaControllerDestroy");
        postConnection = Bind<PostConnectionDelegate>("MaaControllerPostConnection");
        postScreencap = Bind<PostScreencapDelegate>("MaaControllerPostScreencap");
        postTouchDown = Bind<PostTouchDownDelegate>("MaaControllerPostTouchDown");
        postTouchUp = Bind<PostTouchUpDelegate>("MaaControllerPostTouchUp");
        wait = Bind<WaitDelegate>("MaaControllerWait");
        connected = Bind<ConnectedDelegate>("MaaControllerConnected");
        setOption = Bind<SetOptionDelegate>("MaaControllerSetOption");
    }

    private T Bind<T>(string name) where T : Delegate
    {
        IntPtr proc = GetProcAddress(module, name);
        if (proc == IntPtr.Zero) throw new EntryPointNotFoundException(name);
        return Marshal.GetDelegateForFunctionPointer<T>(proc);
    }

    public IntPtr Create(IntPtr hwnd) => create(hwnd, 1UL, 1024UL, 2UL);
    public bool UseRawScreenshotCoordinates(IntPtr controller)
    {
        IntPtr value = Marshal.AllocHGlobal(1);
        try
        {
            Marshal.WriteByte(value, 1);
            return setOption(controller, 3, value, 1) != 0;
        }
        finally { Marshal.FreeHGlobal(value); }
    }
    public void Destroy(IntPtr controller) { if (controller != IntPtr.Zero) destroy(controller); }
    public int Connect(IntPtr controller) => wait(controller, postConnection(controller));
    public int Screencap(IntPtr controller) => wait(controller, postScreencap(controller));
    public int TouchDown(IntPtr controller, int x, int y) => wait(controller, postTouchDown(controller, 0, x, y, 100));
    public int TouchUp(IntPtr controller) => wait(controller, postTouchUp(controller, 0));
    public bool IsConnected(IntPtr controller) => connected(controller) != 0;

    public void Dispose()
    {
        // MaaFramework owns worker threads and subordinate modules. Do not FreeLibrary here;
        // process teardown is the safe unload boundary for this isolated probe.
        module = IntPtr.Zero;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool SetDllDirectory(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr LoadLibrary(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true)] private static extern IntPtr GetProcAddress(IntPtr module, string name);
}
'@

Add-Type -TypeDefinition $source -Language CSharp

$frameworkDll = Join-Path $MaaFrameworkBin 'MaaFramework.dll'
$controlDll = Join-Path $MaaFrameworkBin 'MaaWin32ControlUnit.dll'
if (-not (Test-Path -LiteralPath $frameworkDll -PathType Leaf)) { throw "Missing $frameworkDll" }
if (-not (Test-Path -LiteralPath $controlDll -PathType Leaf)) { throw "Missing $controlDll" }

$window = [AnchoredTouchTestWindow]::new()
$api = $null
$controller = [IntPtr]::Zero
$window.Start()
try {
    Start-Sleep -Milliseconds 250
    $api = [MaaFrameworkApi]::new($MaaFrameworkBin)
    $controller = $api.Create($window.Handle)
    if ($controller -eq [IntPtr]::Zero) { throw 'MaaWin32ControllerCreate returned null.' }
    if (-not $api.UseRawScreenshotCoordinates($controller)) { throw 'Failed to enable raw screenshot coordinates.' }

    $connectStatus = $api.Connect($controller)
    $screencapStatus = $api.Screencap($controller)

    $rectBefore = $window.GetRect()
    $cursorBefore = [AnchoredTouchTestWindow]::GetCursor()
    $foregroundBefore = [AnchoredTouchTestWindow]::ForegroundWindow()

    $clientX = 300
    $clientY = 160
    $hitTestBefore = $window.HitTestClientPoint($clientX, $clientY)
    $downStatus = $api.TouchDown($controller, $clientX, $clientY)
    Start-Sleep -Milliseconds 80
    $upStatus = $api.TouchUp($controller)
    Start-Sleep -Milliseconds 250

    $rectAfter = $window.GetRect()
    $cursorAfter = [AnchoredTouchTestWindow]::GetCursor()
    $foregroundAfter = [AnchoredTouchTestWindow]::ForegroundWindow()
    $messages = $window.Messages

    $result = [ordered]@{
        MaaFrameworkBin = $MaaFrameworkBin
        TestWindow = ('0x{0:X}' -f $window.Handle.ToInt64())
        ConnectStatus = $connectStatus
        Connected = $api.IsConnected($controller)
        ScreencapStatus = $screencapStatus
        TouchDownStatus = $downStatus
        TouchUpStatus = $upStatus
        PointerDownCount = $window.PointerDownCount
        PointerUpCount = $window.PointerUpCount
        HitTestBefore = $hitTestBefore
        CursorBefore = $cursorBefore.ToString()
        CursorAfter = $cursorAfter.ToString()
        CursorUnchanged = $cursorBefore.Equals($cursorAfter)
        WindowRectBefore = $rectBefore.ToString()
        WindowRectAfter = $rectAfter.ToString()
        WindowRectUnchanged = $rectBefore.Equals($rectAfter)
        ForegroundBefore = ('0x{0:X}' -f $foregroundBefore.ToInt64())
        ForegroundAfter = ('0x{0:X}' -f $foregroundAfter.ToInt64())
        ForegroundUnchanged = $foregroundBefore -eq $foregroundAfter
        PointerMessages = @($messages)
    }
    [pscustomobject]$result | ConvertTo-Json -Depth 4

    $passed = $connectStatus -eq 3000 -and $screencapStatus -eq 3000 -and
              $downStatus -eq 3000 -and $upStatus -eq 3000 -and
              $window.PointerDownCount -ge 1 -and $window.PointerUpCount -ge 1 -and
              $cursorBefore.Equals($cursorAfter) -and $rectBefore.Equals($rectAfter) -and
              $foregroundBefore -eq $foregroundAfter
    if (-not $passed) { throw 'AnchoredTouch invariant probe failed; inspect the JSON result above.' }

    if ($KeepWindowOpen) {
        Write-Host 'Probe passed. Press Enter to close the isolated test window.'
        [void](Read-Host)
    }
}
finally {
    if ($api -and $controller -ne [IntPtr]::Zero) { $api.Destroy($controller) }
    if ($api) { $api.Dispose() }
    $window.Dispose()
}
