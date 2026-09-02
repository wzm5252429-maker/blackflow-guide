@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m blackflow.calibrate point --config configs\strategy_first_battle.json --image calibration\window.png
pause

