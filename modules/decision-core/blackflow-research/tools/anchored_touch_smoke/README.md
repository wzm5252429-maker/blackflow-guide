# AnchoredTouch harmless smoke test

This harness creates two disposable local Win32 windows. One receives synthetic
touch and the other remains the foreground sentinel. It directly calls the
MaaFramework v5.13 `ControlUnitAPI` vtable methods `connect`, `screencap`,
`touch_down`, and `touch_up`.

During the injected touch it continuously samples:

- the system cursor position;
- the target window rectangle;
- the foreground window handle.

It does not locate or interact with the game and does not modify the MAA
installation directory.

Run from PowerShell 7:

```powershell
pwsh -NoProfile -File .\Run-AnchoredTouchSmoke.ps1 `
  -ControlDll 'D:\明日方舟\MaaWin32ControlUnit.dll'
```

The selected DLL directory must also contain the dependency set validated by
the installer (`MaaUtils.dll` and `opencv_world4_maa.dll`).
