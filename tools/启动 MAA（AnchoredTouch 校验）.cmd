@echo off
setlocal
chcp 65001 >nul

where pwsh.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell 7 ^(pwsh.exe^) was not found.
    echo Install PowerShell 7 or run the verifier script manually.
    pause
    exit /b 30
)

pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0codex-tools\Start-MaaAnchoredTouch.ps1" -MaaDir "%~dp0"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo AnchoredTouch verification blocked MAA startup. Exit code: %EXIT_CODE%
    pause
)
exit /b %EXIT_CODE%
