@echo off
chcp 65001 >nul
cd /d "%~dp0"
set /p NAME=Detector name (battle_started / victory / defeat / card_name): 
py -m blackflow.calibrate crop --config configs\strategy_first_battle.json --image calibration\window.png --name %NAME%
pause

