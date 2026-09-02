@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Please copy this window and send it to ChatGPT.
  pause
  exit /b 1
)
if not exist configs\strategy_first_battle.json copy /Y configs\strategy_first_battle.example.json configs\strategy_first_battle.json >nul
echo.
echo Dependencies installed successfully.
pause
